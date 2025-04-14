import os
import queue
import sys
import threading
import traceback

import bpy

from .backend import Backend
from .tools import LlamaMeshModelManager, ToolManager
from .utils import get_available_models, get_models_dir

STATUS_CANCELED = "Request canceled by user"
STATUS_DONE = "Agent ran out of steps"


class MESHGEN_OT_Chat(bpy.types.Operator):
    bl_idname = "meshgen.chat"
    bl_label = "Chat"
    bl_description = "Chat with the selected model"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        prefs = bpy.context.preferences.addons[__package__].preferences
        props = context.scene.meshgen_props
        backend = Backend.instance()

        props.history.clear()

        self.temperature = prefs.temperature
        self.prompt = props.prompt
        self.messages = [{"role": "user", "content": self.prompt}]

        self.log_text = bpy.data.texts.get("meshgen log")
        if self.log_text is None:
            self.log_text = bpy.data.texts.new("meshgen log")
        else:
            self.log_text.clear()
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "TEXT_EDITOR":
                        for space in area.spaces:
                            if (
                                space.type == "TEXT_EDITOR"
                                and space.text == self.log_text
                            ):
                                space.show_word_wrap = True
                                space.show_line_numbers = True
                                space.top = 0
                                break

        log_open = False
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "TEXT_EDITOR":
                    continue
                for space in area.spaces:
                    if space.type == "TEXT_EDITOR" and space.text == self.log_text:
                        log_open = True
                        break
                if log_open:
                    break
            if log_open:
                break

        if not log_open:
            bpy.ops.screen.area_split(direction="VERTICAL")
            new_area = context.screen.areas[-1]
            new_area.type = "TEXT_EDITOR"
            new_area.spaces.active.text = self.log_text

        self.log_text.write("\n----- New Chat -----\n")
        self.log_text.write(f"Prompt: {self.prompt}\n")

        if not backend.is_loaded():
            provider_to_model = {
                "LOCAL": prefs.current_model,
                "ollama": prefs.ollama_model_name,
                "huggingface": prefs.huggingface_model_id,
                "anthropic": prefs.anthropic_model_id,
                "openai": prefs.openai_model_id,
            }

            model_name = (
                provider_to_model["LOCAL"]
                if prefs.backend_type == "LOCAL"
                else provider_to_model.get(prefs.llm_provider)
            )
            self.add_event("LOADING", "Loading...", f"Loading {model_name}...")
            try:
                backend.load()
                self.pop_event()
                self.add_event(
                    "LOADING_SUCCESS", model_name, f"Finished loading {model_name}"
                )
            except Exception as e:
                self.pop_event()
                self.add_event(
                    "LOADING_ERROR",
                    str(e),
                    f"Error loading {model_name}:\n{traceback.format_exc()}",
                )
                return {"CANCELLED"}

        self._stop_event = threading.Event()
        self._output_queue = backend.start_chat_completion(
            self.messages, self.temperature, self._stop_event
        )
        props.state = "RUNNING"

        self.add_event("THINKING", None, "Thinking...")

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = context.scene.meshgen_props

        if props.state == "CANCELED" or event.type == "ESC":
            self._stop_event.set()
            self.add_event("CANCELED", STATUS_CANCELED, "Canceled")
            return self.finish(context)

        if event.type == "TIMER" and props.state == "RUNNING":
            try:
                while True:
                    message_type, content = self._output_queue.get_nowait()
                    if message_type == "STEP":
                        thought, full_output = content
                        self.pop_event()
                        self.add_event("STEP", thought, full_output)
                        self.add_event("THINKING", None, "Thinking...")
                        self.redraw(context)
                    elif message_type == "STEP_ERROR":
                        self.pop_event()
                        self.add_event("ERROR", content, f"Error: {content}")
                        self.add_event("THINKING", None, "Thinking...")
                        self.redraw(context)
                    elif message_type == "ERROR":
                        error_msg, full_error = content
                        self.pop_event()
                        self.add_event("ERROR", error_msg, full_error)
                        return self.finish(context)
                    elif message_type == "FINAL_ANSWER":
                        self.pop_event()
                        self.add_event("SUCCESS", content, f"Final Answer: {content}")
                        return self.finish(context)
                    elif message_type == "CANCELED":
                        self.pop_event()
                        self.add_event("CANCELED", STATUS_CANCELED, STATUS_CANCELED)
                        return self.finish(context)
                    elif message_type == "DONE":
                        self.pop_event()
                        self.add_event("ERROR", STATUS_DONE, STATUS_DONE)
                        return self.finish(context)
            except queue.Empty:
                pass

            ToolManager.instance().process_tasks(context)

        return {"PASS_THROUGH"}

    def add_event(self, event_type, message, long_message):
        props = bpy.context.scene.meshgen_props
        event = props.history.add()
        event.type = event_type
        if message:
            event.content = message
        if long_message:
            lines = str(long_message).split("\n")
            for line in lines:
                self.log_text.write(line + "\n")

    def pop_event(self):
        props = bpy.context.scene.meshgen_props
        props.history.remove(len(props.history) - 1)

    def finish(self, context):
        props = context.scene.meshgen_props
        props.state = "READY"
        context.window_manager.event_timer_remove(self._timer)
        self.redraw(context)
        return {"FINISHED"}

    def redraw(self, context):
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class MESHGEN_OT_CancelChat(bpy.types.Operator):
    bl_idname = "meshgen.cancel_chat"
    bl_label = "Cancel chat"
    bl_description = "Cancel the current chat process"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        props = context.scene.meshgen_props
        props.state = "CANCELED"
        return {"FINISHED"}


class MESHGEN_OT_OpenLog(bpy.types.Operator):
    bl_idname = "meshgen.open_log"
    bl_label = "Open log"
    bl_description = "Open the log file"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        log_text = bpy.data.texts.get("meshgen log")
        if log_text is None:
            log_text = bpy.data.texts.new("meshgen log")
        for area in context.screen.areas:
            if area.type == "TEXT_EDITOR":
                area.spaces.active.text = log_text
                return {"FINISHED"}
        bpy.ops.screen.area_split(direction="VERTICAL")
        new_area = context.screen.areas[-1]
        new_area.type = "TEXT_EDITOR"
        new_area.spaces.active.text = log_text
        return {"FINISHED"}


class MESHGEN_OT_DownloadModel(bpy.types.Operator):
    bl_idname = "meshgen.download_model"
    bl_label = "Download model"
    bl_description = "Download model from Hugging Face"
    bl_options = {"REGISTER", "INTERNAL"}

    __annotations__ = {
        "repo_id": bpy.props.StringProperty(name="Repository ID"),
        "filename": bpy.props.StringProperty(name="Filename"),
    }

    _timer = None
    _download_thread = None
    _progress_queue = None

    def execute(self, context):
        if self.filename in get_available_models():
            self.report({"INFO"}, f"{self.filename} already downloaded")
            return {"CANCELLED"}

        import queue
        import re
        import sys
        import threading

        from huggingface_hub import hf_hub_download

        prefs = context.preferences.addons[__package__].preferences
        prefs.downloading = True
        prefs.download_progress = 0

        self._progress_queue = queue.Queue()

        class TqdmCapture:
            def __init__(self, queue, stream):
                self.queue = queue
                self.stream = stream
                self.original_write = stream.write
                self.original_flush = stream.flush

            def write(self, string):
                match = re.search(r"\r.*?(\d+)%", string)
                if match:
                    try:
                        percentage = int(match.group(1))
                        self.queue.put(percentage)
                    except Exception as e:
                        self.queue.put(f"Error parsing progress: {string}, {e}")
                self.original_write(string)
                self.original_flush()

            def flush(self):
                self.original_flush()

        def download_task():
            try:
                old_stderr = sys.stderr
                sys.stderr = TqdmCapture(self._progress_queue, sys.stderr)

                hf_hub_download(
                    self.repo_id,
                    filename=self.filename,
                    local_dir=get_models_dir(),
                )
                prefs = bpy.context.preferences.addons[__package__].preferences
                if prefs.downloading:
                    self._progress_queue.put("finished")
            except InterruptedError:
                pass
            except Exception as e:
                prefs = bpy.context.preferences.addons[__package__].preferences
                if prefs.downloading:
                    self._progress_queue.put(f"Error downloading model: {e}")
            finally:
                sys.stderr = old_stderr

        self._download_thread = threading.Thread(target=download_task)
        self._download_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            prefs = context.preferences.addons[__package__].preferences

            if not prefs.downloading:
                self.report({"INFO"}, "Download canceled")
                self.cleanup(context)
                return {"CANCELLED"}

            while not self._progress_queue.empty():
                item = self._progress_queue.get()
                if isinstance(item, (int, float)):
                    prefs.download_progress = item
                    for window in context.window_manager.windows:
                        for area in window.screen.areas:
                            if area.type == "PREFERENCES":
                                area.tag_redraw()
                elif isinstance(item, str):
                    if item == "finished":
                        prefs.downloading = False
                        if len(get_available_models()) == 1:
                            prefs.current_model = get_available_models()[0]
                        self.report(
                            {"INFO"},
                            f"Successfully downloaded {self.filename} from {self.repo_id}",
                        )
                        self.cleanup(context)
                        return {"FINISHED"}
                    else:
                        self.report({"ERROR"}, item)
                        prefs.downloading = False
                        self.cleanup(context)
                        return {"CANCELLED"}

            return {"RUNNING_MODAL"}
        else:
            return {"PASS_THROUGH"}

    def cleanup(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.downloading = False

        if self._progress_queue:
            try:
                while not self._progress_queue.empty():
                    self._progress_queue.get_nowait()
            except Exception:
                pass
            self._progress_queue = None

        if self._timer:
            try:
                bpy.context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        self._download_thread = None

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "PREFERENCES" or area.type == "VIEW_3D":
                    area.tag_redraw()


class MESHGEN_OT_SelectModel(bpy.types.Operator):
    bl_idname = "meshgen.select_model"
    bl_label = "Select model"
    bl_description = "Select the model to use for generation"
    bl_options = {"REGISTER", "INTERNAL"}

    __annotations__ = {"model": bpy.props.StringProperty(name="Model")}

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.current_model = self.model
        return {"FINISHED"}


class MESHGEN_OT_OpenModelsFolder(bpy.types.Operator):
    bl_idname = "meshgen.open_models_folder"
    bl_label = "Open models folder"
    bl_description = "Open the folder containing downloaded models"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        models_dir = get_models_dir()

        if not models_dir.exists():
            models_dir.mkdir(parents=True)

        try:
            if os.name == "nt":
                os.startfile(models_dir)
            elif os.name == "posix":
                import subprocess

                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.run([opener, models_dir])
        except Exception as e:
            self.report({"ERROR"}, f"Failed to open models folder: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


class MESHGEN_OT_LoadLlamaMesh(bpy.types.Operator):
    bl_idname = "meshgen.load_llama_mesh"
    bl_label = "Load LLaMA-Mesh"
    bl_description = "Load the LLaMA-Mesh model"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        model_manager = LlamaMeshModelManager.instance()
        model_manager.load_model()
        Backend.reset()
        return {"FINISHED"}


class MESHGEN_OT_UnloadLlamaMesh(bpy.types.Operator):
    bl_idname = "meshgen.unload_llama_mesh"
    bl_label = "Unload LLaMA-Mesh"
    bl_description = "Unload the LLaMA-Mesh model"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        model_manager = LlamaMeshModelManager.instance()
        model_manager.unload_model()
        Backend.reset()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(MESHGEN_OT_Chat)
    bpy.utils.register_class(MESHGEN_OT_CancelChat)
    bpy.utils.register_class(MESHGEN_OT_OpenLog)
    bpy.utils.register_class(MESHGEN_OT_DownloadModel)
    bpy.utils.register_class(MESHGEN_OT_SelectModel)
    bpy.utils.register_class(MESHGEN_OT_OpenModelsFolder)
    bpy.utils.register_class(MESHGEN_OT_LoadLlamaMesh)
    bpy.utils.register_class(MESHGEN_OT_UnloadLlamaMesh)


def unregister():
    bpy.utils.unregister_class(MESHGEN_OT_Chat)
    bpy.utils.unregister_class(MESHGEN_OT_CancelChat)
    bpy.utils.unregister_class(MESHGEN_OT_OpenLog)
    bpy.utils.unregister_class(MESHGEN_OT_DownloadModel)
    bpy.utils.unregister_class(MESHGEN_OT_SelectModel)
    bpy.utils.unregister_class(MESHGEN_OT_OpenModelsFolder)
    bpy.utils.unregister_class(MESHGEN_OT_LoadLlamaMesh)
    bpy.utils.unregister_class(MESHGEN_OT_UnloadLlamaMesh)
