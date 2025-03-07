import bpy


class MeshGenProperties(bpy.types.PropertyGroup):
    __annotations__ = {
        "prompt": bpy.props.StringProperty(
            name="Prompt",
            description="Prompt for the mesh generation.",
            default="Create a 3D obj file using the following description: a desk",
        ),
        "is_running": bpy.props.BoolProperty(
            name="Is Running",
            description="Whether the mesh generation is running.",
            default=False,
        ),
        "generated_text": bpy.props.StringProperty(
            name="Generated Text",
            description="The generated text.",
            default="",
        ),
        "cancelled": bpy.props.BoolProperty(
            name="Cancelled",
            description="Whether the mesh generation was cancelled.",
            default=False,
        ),
        "temperature": bpy.props.FloatProperty(
            name="Temperature",
            description="The temperature for the mesh generation.",
            default=0.9,
            min=0.0,
            max=1.0,
        ),
        "vertices_generated": bpy.props.IntProperty(
            name="Vertices Generated",
            description="The number of vertices generated.",
            default=0,
        ),
        "faces_generated": bpy.props.IntProperty(
            name="Faces Generated",
            description="The number of faces generated.",
            default=0,
        ),
        "show_developer_options": bpy.props.BoolProperty(
            name="Show Developer Options",
            description="Whether to show developer options.",
            default=False,
        ),
        "use_ollama_backend": bpy.props.BoolProperty(
            name="Use Ollama for Backend",
            description="Use Ollama for backend processing",
            default=False,
        ),
        "ollama_host": bpy.props.StringProperty(
            name="Ollama Host",
            description="Host address for Ollama backend",
            default="http://localhost:11434",
        ),
    }


def register():
    bpy.utils.register_class(MeshGenProperties)
    bpy.types.Scene.meshgen_props = bpy.props.PointerProperty(type=MeshGenProperties)


def unregister():
    bpy.utils.unregister_class(MeshGenProperties)
