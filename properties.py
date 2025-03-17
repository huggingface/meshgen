import bpy


class MeshGenEvent(bpy.types.PropertyGroup):
    __annotations__ = {
        "type": bpy.props.EnumProperty(
            name="Type",
            description="The type of event",
            items=[
                ("THINKING", "Thinking", "The agent is thinking"),
                ("STEP", "Step", "A step in the agent's thought process"),
                ("ERROR", "Error", "An error occurred"),
                ("CANCELED", "Canceled", "The operation was canceled by the user"),
                ("SUCCESS", "Result", "The result of the operation"),
                ("LOADING", "Loading", "The model is loading"),
                ("LOADING_SUCCESS", "Loaded", "The model loaded successfully"),
                ("LOADING_ERROR", "Error", "The model failed to load"),
            ],
        ),
        "content": bpy.props.StringProperty(
            name="Content",
            description="The content of the event",
            default="",
        ),
    }


class MeshGenProperties(bpy.types.PropertyGroup):
    __annotations__ = {
        "prompt": bpy.props.StringProperty(
            name="Prompt",
            description="Enter a request for the AI agent",
            default="Create a cube",
        ),
        "state": bpy.props.EnumProperty(
            name="State",
            description="The current state of the agent",
            items=[
                ("READY", "Ready", "The agent is ready to process a request"),
                ("LOADING", "Loading", "The agent is loading"),
                ("RUNNING", "Running", "The agent is running"),
                ("CANCELED", "Canceled", "The operation is flagged for cancellation"),
            ],
            default="READY",
        ),
        "history": bpy.props.CollectionProperty(type=MeshGenEvent),
    }


def register():
    bpy.utils.register_class(MeshGenEvent)
    bpy.utils.register_class(MeshGenProperties)
    bpy.types.Scene.meshgen_props = bpy.props.PointerProperty(type=MeshGenProperties)


def unregister():
    bpy.utils.unregister_class(MeshGenEvent)
    bpy.utils.unregister_class(MeshGenProperties)
