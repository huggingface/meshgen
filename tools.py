import queue
import threading
import traceback
import uuid
from typing import Any, List, Optional

import bpy
import mathutils
from smolagents import Tool


class ToolManager:
    """Singleton manager for Blender tools that handles task queueing and execution on the main thread."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ToolManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        prefs = bpy.context.preferences.addons[__package__].preferences

        self._task_queue = queue.Queue()
        self._result_dict = {}
        self._condition = threading.Condition()
        self._tools = [
            GetSceneInfoTool(),
            GetObjectInfoTool(),
            CreateObjectTool(),
            ModifyObjectTool(),
            DeleteObjectTool(),
            SetMaterialTool(),
        ]

        if LlamaMeshModelManager.instance().is_loaded:
            self._tools += [
                LlamaMeshGenerateTool(),
                LlamaMeshDescribeTool(),
            ]

        if prefs.enable_hyper3d:
            self._tools += [Hyper3dGenerateObjectTool()]

    @property
    def tools(self):
        return self._tools

    def add_task(self, task):
        with self._condition:
            self._task_queue.put(task)
            self._condition.notify()

    def get_result(self, task_id):
        with self._condition:
            while task_id not in self._result_dict:
                self._condition.wait()
            return self._result_dict.pop(task_id)

    def process_tasks(self, context):
        try:
            while True:
                task = self._task_queue.get_nowait()
                task_type = task["type"]
                params = task.get("params") or {}

                tool_class = next(
                    (tool for tool in self._tools if tool.name == task_type),
                    None,
                )

                if tool_class and tool_class._main_thread_handler:
                    result = tool_class._main_thread_handler(context, **params)
                    with self._condition:
                        self._result_dict[task["id"]] = result
                        self._condition.notify_all()
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
        except queue.Empty:
            pass

    @classmethod
    def instance(cls):
        return cls()

    @classmethod
    def reset(cls):
        cls._instance = None


def blender_main_thread_handler(main_thread_func):
    """Decorator to associate a main thread function with a tool."""

    def wrapper(tool_class):
        tool_class._main_thread_handler = staticmethod(main_thread_func)
        return tool_class

    return wrapper


class BlenderTool(Tool):
    """
    Base class for all Blender tools.

    Uses task manager to execute tasks in the main blender context.
    """

    _main_thread_handler = None

    def __init__(self):
        super().__init__()
        if not self._main_thread_handler:
            raise ValueError(
                f"No main thread handler set for {self.__class__.__name__}"
            )

    def _execute_task(self, task_type: str, params: dict = None) -> str:
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": task_type,
            "params": params,
        }
        tool_manager = ToolManager.instance()
        tool_manager.add_task(task)
        result = tool_manager.get_result(task_id)
        if result["status"] == "error":
            return f"Error in {task_type}: {result['data']}"
        return result["data"]


def get_aabb(obj):
    """Returns the world-space axis-aligned bounding box (AABB) of an object."""
    if obj.type != "MESH":
        raise TypeError("Object must be a mesh")

    # Get the bounding box corners in local space
    local_bbox_corners = [mathutils.Vector(corner) for corner in obj.bound_box]

    # Convert to world coordinates
    world_bbox_corners = [obj.matrix_world @ corner for corner in local_bbox_corners]

    # Compute axis-aligned min/max coordinates
    min_corner = mathutils.Vector(map(min, zip(*world_bbox_corners)))
    max_corner = mathutils.Vector(map(max, zip(*world_bbox_corners)))

    return [
        [*min_corner],
        [*max_corner],
    ]


def get_scene_info(context: Any):
    try:
        print("Getting scene info...")
        scene_info = {
            "name": context.scene.name,
            "object_count": len(context.scene.objects),
            "objects": [],
            "materials_count": len(bpy.data.materials),
        }

        for i, obj in enumerate(context.scene.objects):
            if i >= 10:
                break

            obj_info = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
            }
            scene_info["objects"].append(obj_info)

        print(f"Scene info collected: {len(scene_info['objects'])} objects")
        return {"status": "success", "data": scene_info}
    except Exception as e:
        print(f"Error in get_scene_info: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(get_scene_info)
class GetSceneInfoTool(BlenderTool):
    name = "get_scene_info"
    description = """
    This is a tool that gets detailed information about the current Blender scene.

    It returns the scene name, object count, a list of objects with their names, types, and locations, and the number of materials in the scene.

    Details are only provided for the first 10 objects, but the total object count is provided.

    The output is a dictionary with the following structure:
    {
        "name": "Scene",
        "object_count": 3,
        "objects": [
            {
                "name": "Cube",
                "type": "MESH",
                "location": [1.00, 2.00, 3.00]
            },
            {
                "name": "Sphere",
                "type": "MESH",
                "location": [0.00, 0.00, 0.00]
            },
            {
                "name": "Camera",
                "type": "CAMERA",
                "location": [0.00, 0.00, 0.00]
            }
        ],
        "materials_count": 2
    }
    """
    inputs = {}
    output_type = "object"

    def forward(self):
        return self._execute_task("get_scene_info")


def get_object_info(context: Any, name: str):
    try:
        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object with name {name} not found")

        obj_info = {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [
                obj.rotation_euler.x,
                obj.rotation_euler.y,
                obj.rotation_euler.z,
            ],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            "visible": obj.visible_get(),
            "materials": [],
        }

        if obj.type == "MESH":
            bounding_box = get_aabb(obj)
            obj_info["world_bounding_box"] = bounding_box

        for slot in obj.material_slots:
            if slot.material:
                obj_info["materials"].append(slot.material.name)

        if obj.type == "MESH" and obj.data:
            mesh = obj.data
            obj_info["mesh"] = {
                "vertices": len(mesh.vertices),
                "edges": len(mesh.edges),
                "polygons": len(mesh.polygons),
            }

        return {"status": "success", "data": obj_info}
    except Exception as e:
        print(f"Error in get_object_info: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(get_object_info)
class GetObjectInfoTool(BlenderTool):
    name = "get_object_info"
    description = """
    This is a tool that gets detailed information about a specific object in the Blender scene.

    It returns the object name, type, location, rotation, scale, visibility, materials, and world bounding box (if applicable).

    The output is a dictionary with the following structure:
    {
        "name": "Cube",
        "type": "MESH",
        "location": [1.00, 2.00, 3.00],
        "rotation": [0.00, 0.00, 0.00],
        "scale": [1.00, 1.00, 1.00],
        "visible": True,
        "materials": ["Material1"],
        "world_bounding_box": [1.00, 2.00, 3.00, 1.00, 1.00, 1.00]
    }
    """
    inputs = {
        "name": {
            "type": "string",
            "description": "Name of the object to get information about",
        }
    }
    output_type = "object"

    def forward(self, name: str):
        return self._execute_task("get_object_info", {"name": name})


def create_object(
    context: Any,
    type: str,
    name: Optional[str] = None,
    location: List[float] = None,
    rotation: List[float] = None,
    scale: List[float] = None,
    align: Optional[str] = None,
    major_segments: Optional[int] = None,
    minor_segments: Optional[int] = None,
    mode: Optional[str] = None,
    major_radius: Optional[float] = None,
    minor_radius: Optional[float] = None,
    abso_major_rad: Optional[float] = None,
    abso_minor_rad: Optional[float] = None,
    generate_uvs: Optional[bool] = None,
):
    try:
        view_area_3d = next(
            (area for area in context.screen.areas if area.type == "VIEW_3D"), None
        )
        if view_area_3d is None:
            raise RuntimeError("View 3D area not found")

        override = context.copy()
        override["area"] = view_area_3d

        with context.temp_override(**override):
            # Deselect all objects first
            bpy.ops.object.select_all(action="DESELECT")

            location = location or [0, 0, 0]
            rotation = rotation or [0, 0, 0]
            scale = scale or [1, 1, 1]

            # Create the object based on type
            if type == "CUBE":
                bpy.ops.mesh.primitive_cube_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "SPHERE":
                bpy.ops.mesh.primitive_uv_sphere_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "CYLINDER":
                bpy.ops.mesh.primitive_cylinder_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "PLANE":
                bpy.ops.mesh.primitive_plane_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "CONE":
                bpy.ops.mesh.primitive_cone_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "TORUS":
                bpy.ops.mesh.primitive_torus_add(
                    align=align,
                    location=location,
                    rotation=rotation,
                    major_segments=major_segments,
                    minor_segments=minor_segments,
                    mode=mode,
                    major_radius=major_radius,
                    minor_radius=minor_radius,
                    abso_major_rad=abso_major_rad,
                    abso_minor_rad=abso_minor_rad,
                    generate_uvs=generate_uvs,
                )
            elif type == "EMPTY":
                bpy.ops.object.empty_add(
                    location=location, rotation=rotation, scale=scale
                )
            elif type == "CAMERA":
                bpy.ops.object.camera_add(location=location, rotation=rotation)
            elif type == "LIGHT":
                bpy.ops.object.light_add(
                    type="POINT",
                    location=location,
                    rotation=rotation,
                    scale=scale,
                )
            else:
                raise ValueError(f"Unsupported object type: {type}")

            # Force update the view layer
            bpy.context.view_layer.update()

            # Get the active object (which should be our newly created object)
            obj = bpy.context.view_layer.objects.active

            # If we don't have an active object, something went wrong
            if obj is None:
                raise RuntimeError("Failed to create object - no active object")

            # Make sure it's selected
            obj.select_set(True)

            # Rename if name is provided
            if name:
                obj.name = name
                if obj.data:
                    obj.data.name = name

            # Patch for PLANE: scale don't work with bpy.ops.mesh.primitive_plane_add()
            if type in {"PLANE"}:
                obj.scale = scale

            # Return the object info
            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [
                    obj.rotation_euler.x,
                    obj.rotation_euler.y,
                    obj.rotation_euler.z,
                ],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
            }

            if obj.type == "MESH":
                bounding_box = get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error in create_object: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(create_object)
class CreateObjectTool(BlenderTool):
    name = "create_object"
    description = """
    This is a tool that creates a new object in the Blender scene.

    It returns the created object name, type, location, rotation, scale, and world bounding box (if applicable).

    The output is a dictionary with the following structure:
    {
        "name": "Cube",
        "type": "MESH",
        "location": [1.00, 2.00, 3.00],
        "rotation": [0.00, 0.00, 0.00],
        "scale": [1.00, 1.00, 1.00],
        "world_bounding_box": [1.00, 2.00, 3.00, 1.00, 1.00, 1.00]
    }
    """
    inputs = {
        "type": {
            "type": "string",
            "description": "Object type (CUBE, SPHERE, CYLINDER, PLANE, CONE, TORUS, EMPTY, CAMERA, LIGHT)",
        },
        "name": {
            "type": "string",
            "description": "Optional name for the object",
            "nullable": True,
        },
        "location": {
            "type": "array",
            "description": "Optional [x, y, z] location coordinates",
            "nullable": True,
        },
        "rotation": {
            "type": "array",
            "description": "Optional [x, y, z] rotation in radians",
            "nullable": True,
        },
        "scale": {
            "type": "array",
            "description": "Optional [x, y, z] scale factors (not used for TORUS)",
            "nullable": True,
        },
        "align": {
            "type": "string",
            "description": "How to align the torus ('WORLD', 'VIEW', or 'CURSOR')",
            "nullable": True,
        },
        "major_segments": {
            "type": "integer",
            "description": "Number of segments for the main ring",
            "nullable": True,
        },
        "minor_segments": {
            "type": "integer",
            "description": "Number of segments for the cross-section",
            "nullable": True,
        },
        "mode": {
            "type": "string",
            "description": "Dimension mode ('MAJOR_MINOR' or 'EXT_INT')",
            "nullable": True,
        },
        "major_radius": {
            "type": "number",
            "description": "Radius from the origin to the center of the cross sections",
            "nullable": True,
        },
        "minor_radius": {
            "type": "number",
            "description": "Radius of the torus' cross section",
            "nullable": True,
        },
        "abso_major_rad": {
            "type": "number",
            "description": "Total exterior radius of the torus",
            "nullable": True,
        },
        "abso_minor_rad": {
            "type": "number",
            "description": "Total interior radius of the torus",
            "nullable": True,
        },
        "generate_uvs": {
            "type": "boolean",
            "description": "Whether to generate a default UV map",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(
        self,
        type: str,
        name: Optional[str] = None,
        location: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        align: Optional[str] = "WORLD",
        major_segments: Optional[int] = 48,
        minor_segments: Optional[int] = 12,
        mode: Optional[str] = "MAJOR_MINOR",
        major_radius: Optional[float] = 1.0,
        minor_radius: Optional[float] = 0.25,
        abso_major_rad: Optional[float] = 1.25,
        abso_minor_rad: Optional[float] = 1.25,
        generate_uvs: Optional[bool] = True,
    ):
        location = location or [0, 0, 0]
        rotation = rotation or [0, 0, 0]
        scale = scale or [1, 1, 1]

        params = {
            "type": type,
            "location": location,
            "rotation": rotation,
            "scale": scale,
        }

        if name:
            params["name"] = name

        if type == "TORUS":
            params.update(
                {
                    "align": align,
                    "major_segments": major_segments,
                    "minor_segments": minor_segments,
                    "mode": mode,
                    "major_radius": major_radius,
                    "minor_radius": minor_radius,
                    "abso_major_rad": abso_major_rad,
                    "abso_minor_rad": abso_minor_rad,
                    "generate_uvs": generate_uvs,
                }
            )

        return self._execute_task("create_object", params)


def modify_object(
    context: Any,
    name: str,
    location: Optional[List[float]] = None,
    rotation: Optional[List[float]] = None,
    scale: Optional[List[float]] = None,
    visible: Optional[bool] = None,
):
    try:
        view_area_3d = next(
            (area for area in context.screen.areas if area.type == "VIEW_3D"), None
        )
        if view_area_3d is None:
            raise RuntimeError("View 3D area not found")

        override = context.copy()
        override["area"] = view_area_3d

        with context.temp_override(**override):
            obj = bpy.data.objects.get(name)
            if not obj:
                raise ValueError(f"Object with name {name} not found")

            if location is not None:
                obj.location = location
            if rotation is not None:
                obj.rotation_euler = rotation
            if scale is not None:
                obj.scale = scale
            if visible is not None:
                obj.visible_set(visible)

            result = {
                "name": obj.name,
                "type": obj.type,
                "location": [obj.location.x, obj.location.y, obj.location.z],
                "rotation": [
                    obj.rotation_euler.x,
                    obj.rotation_euler.y,
                    obj.rotation_euler.z,
                ],
                "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
                "visible": obj.visible_get(),
            }

            if obj.type == "MESH":
                bounding_box = get_aabb(obj)
                result["world_bounding_box"] = bounding_box

            return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error in modify_object: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(modify_object)
class ModifyObjectTool(BlenderTool):
    name = "modify_object"
    description = """
    This is a tool that modifies an existing object in the Blender scene.

    It can modify the object's location, rotation, scale, and visibility.
    
    It returns the modified object name, type, location, rotation, scale, and visibility.
    
    The output is a dictionary with the following structure:
    {
        "name": "Cube",
        "type": "MESH",
        "location": [1.00, 2.00, 3.00],
        "rotation": [0.00, 0.00, 0.00],
        "scale": [1.00, 1.00, 1.00],
        "visible": True
    }
    """
    inputs = {
        "name": {
            "type": "string",
            "description": "Name of the object to modify",
        },
        "location": {
            "type": "array",
            "description": "Optional [x, y, z] location coordinates",
            "nullable": True,
        },
        "rotation": {
            "type": "array",
            "description": "Optional [x, y, z] rotation in radians",
            "nullable": True,
        },
        "scale": {
            "type": "array",
            "description": "Optional [x, y, z] scale factors",
            "nullable": True,
        },
        "visible": {
            "type": "boolean",
            "description": "Optional visibility of the object",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(
        self,
        name: str,
        location: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        visible: Optional[bool] = None,
    ):
        params = {
            "name": name,
            "location": location,
            "rotation": rotation,
            "scale": scale,
            "visible": visible,
        }
        return self._execute_task("modify_object", params)


def delete_object(context: Any, name: str) -> dict:
    try:
        view_area_3d = next(
            (area for area in context.screen.areas if area.type == "VIEW_3D"), None
        )
        if view_area_3d is None:
            raise RuntimeError("View 3D area not found")

        override = context.copy()
        override["area"] = view_area_3d

        with context.temp_override(**override):
            obj = bpy.data.objects.get(name)
            if not obj:
                raise ValueError(f"Object with name {name} not found")

            bpy.data.objects.remove(obj, do_unlink=True)
            return {"status": "success", "data": name}
    except Exception as e:
        print(f"Error in delete_object: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(delete_object)
class DeleteObjectTool(BlenderTool):
    name = "delete_object"
    description = """
    This is a tool that deletes an existing object from the Blender scene.

    It returns the deleted object name.
    """
    inputs = {
        "name": {
            "type": "string",
            "description": "Name of the object to delete",
        }
    }
    output_type = "string"

    def forward(self, name: str):
        return self._execute_task("delete_object", {"name": name})


def set_material(
    context: Any,
    object_name: str,
    material_name: Optional[str] = None,
    create_if_missing: Optional[bool] = True,
    color: Optional[List[float]] = None,
):
    try:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            raise ValueError(f"Object with name {object_name} not found")

        if not hasattr(obj, "data") or not hasattr(obj.data, "materials"):
            raise ValueError(f"Object {object_name} cannot accept materials")

        # Create or get material
        if material_name:
            mat = bpy.data.materials.get(material_name)
            if not mat and create_if_missing:
                mat = bpy.data.materials.new(name=material_name)
                print(f"Created new material: {material_name}")
        else:
            mat_name = f"{object_name}_material"
            mat = bpy.data.materials.get(mat_name)
            if not mat:
                mat = bpy.data.materials.new(name=mat_name)
            material_name = mat_name
            print(f"Using material: {material_name}")

        # Set up material nodes if needed
        if mat:
            if not mat.use_nodes:
                mat.use_nodes = True

            # Get or create Principled BSDF
            principled = mat.node_tree.nodes.get("Principled BSDF")
            if not principled:
                principled = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
                output = mat.node_tree.nodes.new("Material Output")
                if not output:
                    output = mat.node_tree.nodes.new("Material Output")
                if not principled.outputs[0].links:
                    mat.node_tree.links.new(principled.outputs[0], output.inputs[0])

            # Set color if provided
            if color and len(color) >= 3:
                principled.inputs["Base Color"].default_value = (
                    color[0],
                    color[1],
                    color[2],
                    1.0 if len(color) < 4 else color[3],
                )
                print(f"Set material color to {color}")

        if mat:
            if not obj.data.materials:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat

            print(f"Assigned material {mat.name} to {object_name}")

            result = {
                "object": object_name,
                "material": mat.name,
                "color": color if color else None,
            }

            return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error in set_material: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(set_material)
class SetMaterialTool(BlenderTool):
    name = "set_material"
    description = """
    Set or create a material on an object.

    Returns the object name, material name, and color.

    The output is a dictionary with the following structure:
    {
        "object": "Cube",
        "material": "Material1",
        "color": [1.00, 0.00, 0.00, 1.00]
    }
    """
    inputs = {
        "object_name": {
            "type": "string",
            "description": "Name of the object to set the material on",
        },
        "material_name": {
            "type": "string",
            "description": "Optional name of the material to use or create",
            "nullable": True,
        },
        "create_if_missing": {
            "type": "boolean",
            "description": "Whether to create a new material if it doesn't exist",
            "nullable": True,
        },
        "color": {
            "type": "array",
            "description": "Optional [r, g, b, a] color values (0-1)",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(
        self,
        object_name: str,
        material_name: Optional[str] = None,
        create_if_missing: Optional[bool] = True,
        color: Optional[List[float]] = None,
    ):
        params = {
            "object_name": object_name,
            "material_name": material_name,
            "create_if_missing": create_if_missing,
            "color": color,
        }
        return self._execute_task("set_material", params)


class LlamaMeshModelManager:
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LlamaMeshModelManager, cls).__new__(cls)

        return cls._instance

    @property
    def is_loaded(self):
        return self._model is not None

    def load_model(self):
        if self._model is not None:
            return

        from llama_cpp import Llama

        repo_id = "bartowski/LLaMA-Mesh-GGUF"
        filename = "LLaMA-Mesh-Q4_K_M.gguf"

        self._model = Llama.from_pretrained(
            repo_id=repo_id, filename=filename, n_gpu_layers=-1, n_ctx=8192
        )

    def unload_model(self):
        if self._model is None:
            return

        import gc

        del self._model
        gc.collect()

        self._model = None

    def get_model(self):
        if not self.is_loaded:
            raise RuntimeError(
                "LLaMA-Mesh is not loaded. You are not able to use this tool."
            )
        return self._model

    @classmethod
    def instance(cls):
        return cls()


def llama_mesh_generate(
    context: Any, object_name: str, object_description: str, temperature: float = 0.5
):
    try:
        import bmesh

        view_area_3d = next(
            (area for area in context.screen.areas if area.type == "VIEW_3D"), None
        )
        if view_area_3d is None:
            raise RuntimeError("View 3D area not found")

        override = context.copy()
        override["area"] = view_area_3d

        with context.temp_override(**override):
            model_manager = LlamaMeshModelManager.instance()
            model = model_manager.get_model()

            messages = [
                {
                    "role": "system",
                    "content": "You are a 3D mesh generator. You will be given a description of a 3D object and generate an obj file.",
                },
                {
                    "role": "user",
                    "content": f"Generate an obj file for the following description: {object_description}",
                },
            ]

            response = model.create_chat_completion(
                messages=messages, stream=True, temperature=temperature
            )

            mesh_data = bpy.data.meshes.new(object_name)
            mesh_obj = bpy.data.objects.new(object_name, mesh_data)
            context.collection.objects.link(mesh_obj)
            bm = bmesh.new()

            line_buffer = ""

            def add_vertex(x, y, z):
                try:
                    bm.verts.new((x, y, z))
                    bm.verts.ensure_lookup_table()
                    return True
                except ValueError as e:
                    print(f"Error adding vertex: ({x}, {y}, {z}): {e}")
                    return False

            def add_face(a, b, c):
                try:
                    bm.faces.new((bm.verts[a - 1], bm.verts[b - 1], bm.verts[c - 1]))
                    bm.faces.ensure_lookup_table()
                    return True
                except ValueError as e:
                    print(f"Error adding face: ({a}, {b}, {c}): {e}")
                    return False

            def process_line(line):
                print(line)
                line = line.strip()

                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) == 4:
                        try:
                            x, y, z = map(int, parts[1:])
                            scale = 1 / 64.0
                            add_vertex(
                                x * scale - 0.5, z * scale - 0.5, y * scale - 0.5
                            )
                        except ValueError:
                            pass
                elif line.startswith("f "):
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            a, b, c = map(int, parts[1:])
                            add_face(a, b, c)
                        except ValueError:
                            pass

            for chunk in response:
                try:
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        line_buffer += content

                        if "\n" in line_buffer:
                            lines = line_buffer.split("\n")
                            for line in lines[:-1]:
                                process_line(line)
                            line_buffer = lines[-1]
                except Exception as e:
                    print(f"Error in process_chunk: {str(e)}")
                    pass

            if line_buffer:
                process_line(line_buffer)

            bm.to_mesh(mesh_data)
            mesh_data.update()
            context.view_layer.update()
            bm.free()

            return {"status": "success", "data": object_name}
    except Exception as e:
        print(f"Error in generate_mesh: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(llama_mesh_generate)
class LlamaMeshGenerateTool(BlenderTool):
    name = "llama_mesh_generate"
    description = """
    Use LLaMA-Mesh to generate a 3D mesh from a description.

    This tool is capable of generating low-resolution meshes with a small number of vertices.

    Returns the name of the generated object.
    """
    inputs = {
        "object_name": {
            "type": "string",
            "description": "Name for the generated object",
        },
        "object_description": {
            "type": "string",
            "description": "Description of the mesh to generate",
        },
        "temperature": {
            "type": "number",
            "description": "Temperature for the model from 0 to 1, where 0 is deterministic and 1 is random",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        object_name: str,
        object_description: str,
        temperature: Optional[float] = 0.5,
    ):
        params = {
            "object_name": object_name,
            "object_description": object_description,
            "temperature": temperature,
        }
        return self._execute_task("llama_mesh_generate", params)


def llama_mesh_describe(context: Any, name: str):
    try:
        model_manager = LlamaMeshModelManager.instance()
        model = model_manager.get_model()

        obj = bpy.data.objects.get(name)
        if not obj:
            raise ValueError(f"Object with name {name} not found")

        if not obj.type == "MESH":
            raise ValueError(f"Object with name {name} is not a mesh")

        bounding_box = get_aabb(obj)
        mesh = obj.data

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")

        min_coords = bounding_box[0]
        max_coords = bounding_box[1]

        ranges = [max_coords[i] - min_coords[i] for i in range(3)]
        max_range = max(ranges)

        vertices_with_indices = []
        for i, vertex in enumerate(mesh.vertices):
            world_vertex = obj.matrix_world @ vertex.co
            quantized_vertex = [
                int((world_vertex[0] - min_coords[0]) / max_range * 63),
                int((world_vertex[2] - min_coords[2]) / max_range * 63),
                int((world_vertex[1] - min_coords[1]) / max_range * 63),
            ]
            vertices_with_indices.append((quantized_vertex, i))

        vertices_with_indices.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

        old_to_new_indices = {
            old_idx: new_idx
            for new_idx, (_, old_idx) in enumerate(vertices_with_indices)
        }

        obj_lines = []

        for vertex in vertices_with_indices:
            v = vertex[0]
            obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")

        faces = []

        for face in mesh.polygons:
            old_indices = list(face.vertices)
            new_indices = [old_to_new_indices[old_idx] for old_idx in old_indices]
            min_pos = new_indices.index(min(new_indices))
            new_indices = new_indices[min_pos:] + new_indices[:min_pos]
            faces.append(new_indices)

        faces.sort(key=lambda x: (x[0], x[1], x[2]))

        for face in faces:
            obj_lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")

        obj_data = "\n".join(obj_lines)

        messages = [
            {
                "role": "system",
                "content": """You are a knowledgeable, efficient, and direct AI assistant that can read 3D obj file data.
                Provide concise answers, focusing only on key information needed.
                """,
            },
            {
                "role": "user",
                "content": f"What is this object?\n{obj_data}",
            },
        ]

        response = model.create_chat_completion(
            messages=messages, stream=False, temperature=0.5
        )

        response = response["choices"][0]["message"]["content"]

        return {"status": "success", "data": response}

    except Exception as e:
        print(f"Error in llama_mesh_understand: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(llama_mesh_describe)
class LlamaMeshDescribeTool(BlenderTool):
    name = "llama_mesh_describe"
    description = """
    Use LLaMA-Mesh to describe a 3D object given its mesh data.

    Only accepts objects with fewer than 800 vertices. Use `get_object_info` to check the number of vertices.

    Returns a string describing what the object is.
    """
    inputs = {
        "name": {
            "type": "string",
            "description": "Name of the object to describe",
        },
    }
    output_type = "string"

    def forward(self, name: str):
        params = {"name": name}
        return self._execute_task("llama_mesh_describe", params)


def _clean_imported_glb(filepath, mesh_name=None):
    existing_objects = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    bpy.context.view_layer.update()

    imported_objects = list(set(bpy.data.objects) - existing_objects)

    if not imported_objects:
        raise RuntimeError("Error: No objects were imported.")

    mesh_obj = None

    if len(imported_objects) == 1 and imported_objects[0].type == "MESH":
        mesh_obj = imported_objects[0]
        print("Single mesh imported, no cleanup needed.")
    else:
        parent_obj = imported_objects[0]
        if parent_obj.type == "EMPTY" and len(parent_obj.children) == 1:
            potential_mesh = parent_obj.children[0]
            if potential_mesh.type == "MESH":
                print("GLB structure confirmed: Empty node with one mesh child.")
                potential_mesh.parent = None

                bpy.data.objects.remove(parent_obj)
                print("Removed empty node, keeping only the mesh.")
                mesh_obj = potential_mesh
            else:
                raise RuntimeError("Error: Child is not a mesh object.")
        else:
            raise RuntimeError(
                "Error: Expected an empty node with one mesh child or a single mesh object."
            )

    try:
        if mesh_obj and mesh_obj.name is not None and mesh_name:
            mesh_obj.name = mesh_name
            if mesh_obj.data.name is not None:
                mesh_obj.data.name = mesh_name
            print(f"Mesh renamed to: {mesh_name}")
    except Exception:
        print("Having issue with renaming, give up renaming.")

    return mesh_obj


def hyper3d_generate_object(context: Any, object_description: str, mesh_name: str):
    try:
        import os
        import tempfile
        import time

        import requests

        prefs = bpy.context.preferences.addons[__package__].preferences
        api_key = prefs.hyper3d_api_key

        files = [
            ("mesh_mode", (None, "Raw")),
            ("prompt", (None, object_description)),
        ]

        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/rodin",
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
        )

        data = response.json()

        succeed = data.get("submit_time", False)
        if not succeed:
            raise RuntimeError("Failed to create generate job")

        task_uuid = data["uuid"]
        subscription_key = data["jobs"]["subscription_key"]

        start_time = time.time()
        max_wait_time = 300  # 5 minutes

        print(f"Generation started. Task UUID: {task_uuid}")
        print(f"Waiting up to {max_wait_time} seconds for generation to complete...")

        while True:
            if time.time() - start_time > max_wait_time:
                raise RuntimeError(
                    f"Generation timed out after {max_wait_time} seconds"
                )

            response = requests.post(
                "https://hyperhuman.deemos.com/api/v2/status",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"subscription_key": subscription_key},
            )

            data = response.json()
            status_list = [i["status"] for i in data["jobs"]]
            if all(status == "Done" for status in status_list):
                break

            time.sleep(2)

        print("Generation completed. Downloading result...")

        response = requests.post(
            "https://hyperhuman.deemos.com/api/v2/download",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"task_uuid": task_uuid},
        )

        data = response.json()
        temp_file = None

        for i in data["list"]:
            if i["name"].endswith(".glb"):
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=task_uuid,
                    suffix=".glb",
                )

                try:
                    response = requests.get(i["url"], stream=True)
                    response.raise_for_status()

                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)

                    temp_file.close()
                except Exception as e:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    raise e

        obj = _clean_imported_glb(temp_file.name, mesh_name)
        result = {
            "name": obj.name,
            "type": obj.type,
            "location": [obj.location.x, obj.location.y, obj.location.z],
            "rotation": [
                obj.rotation_euler.x,
                obj.rotation_euler.y,
                obj.rotation_euler.z,
            ],
            "scale": [obj.scale.x, obj.scale.y, obj.scale.z],
        }

        if obj.type == "MESH":
            bounding_box = get_aabb(obj)
            result["world_bounding_box"] = bounding_box

        return {"status": "success", "data": result}
    except Exception as e:
        print(f"Error in hyper3d_generate: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "data": str(e)}


@blender_main_thread_handler(hyper3d_generate_object)
class Hyper3dGenerateObjectTool(BlenderTool):
    name = "hyper3d_generate_object"
    description = """
    Generate a 3D asset using Hyper3D by giving a description of the desired asset, then import the result into Blender.
    The 3D asset has built-in materials.    
    The generated model has a normalized size, so re-scaling after generation may be useful.

    Hyper3D is good at generating 3D models for a single item.
    Don't try to:
    1. Generate the whole scene at once.
    2. Generate terrain.
    3. Generate parts of the item separately and put them together.
    """
    inputs = {
        "object_description": {
            "type": "string",
            "description": "A short description of the object to generate",
        },
        "mesh_name": {
            "type": "string",
            "description": "Name of the mesh to generate",
        },
    }
    output_type = "object"

    def forward(self, object_description: str, mesh_name: str):
        params = {"object_description": object_description, "mesh_name": mesh_name}
        return self._execute_task("hyper3d_generate_object", params)
