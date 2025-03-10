import queue
import sys
import threading
import traceback

import bmesh
import bpy

from .backend import LiteLLMBackend, LocalLLM


class Generator:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Generator, cls).__new__(cls)
            cls._instance._backend = None
        return cls._instance

    @property
    def backend(self):
        prefs = bpy.context.preferences.addons[__package__].preferences

        if self._backend is None or (
            (
                prefs.backend_type == "REMOTE"
                and not isinstance(self._backend, LiteLLMBackend)
            )
            or (
                prefs.backend_type == "LOCAL"
                and not isinstance(self._backend, LocalLLM)
            )
        ):
            if prefs.backend_type == "LOCAL":
                self._backend = LocalLLM()
            else:
                self._backend = LiteLLMBackend()

        return self._backend

    def create_chat_completion(self, messages, stream=True, temperature=0.9):
        if not self.is_backend_valid():
            raise ValueError("Backend is not valid")

        if not self.is_backend_loaded():
            self.backend.load()

        return self.backend.create_chat_completion(
            messages, stream=stream, temperature=temperature
        )

    def parse_chunk(self, chunk):
        return self.backend.parse_chunk(chunk)

    def is_backend_valid(self):
        return self.backend is not None and self.backend.is_valid()

    def is_backend_loaded(self):
        return self.backend is not None and self.backend.is_loaded()

    @classmethod
    def instance(cls):
        return cls()


class MESHGEN_OT_GenerateMesh(bpy.types.Operator):
    bl_idname = "meshgen.generate"
    bl_label = "Generate Mesh"
    bl_description = "Generate a 3D mesh using the selected model"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        props = context.scene.meshgen_props
        props.cancelled = False
        props.vertices_generated = 0
        props.faces_generated = 0

        self.mesh_data = bpy.data.meshes.new("GeneratedMesh")
        self.mesh_obj = bpy.data.objects.new("GeneratedMesh", self.mesh_data)
        context.collection.objects.link(self.mesh_obj)
        self.bmesh = bmesh.new()

        generator = Generator.instance()

        if not generator.is_backend_valid():
            self.report(
                {"ERROR"}, "Backend configuration is not valid. Check preferences."
            )
            return {"CANCELLED"}

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can generate 3D obj files.",
            },
            {"role": "user", "content": props.prompt},
        ]
        self.generated_text = ""
        self.line_buffer = ""

        try:
            self._iterator = generator.create_chat_completion(
                messages=messages, stream=True, temperature=props.temperature
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to create chat completion: {str(e)}")
            return {"CANCELLED"}

        props.is_running = True
        self._queue = queue.Queue()

        def run_in_thread():
            try:
                for chunk in self._iterator:
                    if props.cancelled:
                        return
                    self._queue.put(chunk)
                self._queue.put(None)
            except Exception as e:
                print(e, file=sys.stderr)
                traceback.print_exc()
                self._queue.put(None)

        self._thread = threading.Thread(target=run_in_thread)
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def redraw(self, context):
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

    def modal(self, context, event):
        props = context.scene.meshgen_props
        generator = Generator.instance()

        if event.type == "TIMER":
            new_tokens = False
            try:
                while not self._queue.empty():
                    chunk = self._queue.get_nowait()
                    if chunk is None:
                        break

                    content = generator.parse_chunk(chunk)
                    if content is None:
                        continue

                    self.generated_text += content
                    self.line_buffer += content
                    props.generated_text = self.generated_text
                    new_tokens = True

                    if "\n" in self.line_buffer:
                        lines = self.line_buffer.split("\n")
                        for line in lines[:-1]:
                            self.process_line(line.strip(), context)
                        self.line_buffer = lines[-1]
            except Exception as e:
                print(e, file=sys.stderr)
                traceback.print_exc()
                props.is_running = False
                context.window_manager.event_timer_remove(self._timer)
                self.bmesh.free()
                return {"CANCELLED"}

            if new_tokens:
                self.redraw(context)

            if not self._thread.is_alive() and self._queue.empty():
                if self.line_buffer:
                    self.process_line(self.line_buffer.strip(), context)
                    self.update_mesh(context)
                props.is_running = False
                context.window_manager.event_timer_remove(self._timer)
                self.bmesh.free()
                self.redraw(context)
                return {"FINISHED"}

        if props.cancelled:
            props.is_running = False
            context.window_manager.event_timer_remove(self._timer)
            self.bmesh.free()
            self.redraw(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def process_line(self, line, context):
        print(line)
        if line.startswith("v "):
            parts = line.split()
            if len(parts) == 4:
                try:
                    x, y, z = map(float, parts[1:])
                    self.add_vertex(x, y, z, context)
                except ValueError:
                    print(f"Invalid vertex line: {line}", file=sys.stderr)
                    traceback.print_exc()
        elif line.startswith("f "):
            parts = line.split()
            if len(parts) == 4:
                try:
                    a, b, c = map(int, parts[1:])
                    self.add_face(a, b, c, context)
                except ValueError:
                    print(f"Invalid face line: {line}", file=sys.stderr)
                    traceback.print_exc()

    def add_vertex(self, x, y, z, context):
        self.bmesh.verts.new((x / 64.0, y / 64.0, z / 64.0))
        self.bmesh.verts.ensure_lookup_table()
        context.scene.meshgen_props.vertices_generated += 1
        self.update_mesh(context)

    def add_face(self, a, b, c, context):
        try:
            self.bmesh.faces.new(
                (
                    self.bmesh.verts[a - 1],
                    self.bmesh.verts[b - 1],
                    self.bmesh.verts[c - 1],
                )
            )
            self.bmesh.faces.ensure_lookup_table()
            context.scene.meshgen_props.faces_generated += 1
            self.update_mesh(context)
        except IndexError:
            print(f"Indices out of range: {a}, {b}, {c}", file=sys.stderr)
        except ValueError:
            print(f"Face already exists: {a}, {b}, {c}", file=sys.stderr)

    def update_mesh(self, context):
        self.bmesh.to_mesh(self.mesh_data)
        self.mesh_data.update()


class MESHGEN_OT_CancelGeneration(bpy.types.Operator):
    bl_idname = "meshgen.cancel"
    bl_label = "Cancel Generation"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        props = context.scene.meshgen_props
        props.cancelled = True
        return {"FINISHED"}


def register():
    bpy.utils.register_class(MESHGEN_OT_GenerateMesh)
    bpy.utils.register_class(MESHGEN_OT_CancelGeneration)


def unregister():
    bpy.utils.unregister_class(MESHGEN_OT_GenerateMesh)
    bpy.utils.unregister_class(MESHGEN_OT_CancelGeneration)
