import bpy

from .backend import Backend
from .tools import LlamaMeshModelManager
from .utils import get_available_models


def get_downloaded_models(self, context):
    items = []
    for model in get_available_models():
        items.append((model, model, f"Use {model} for local inference"))

    if not items:
        items.append(("", "No models downloaded", "Download a model first"))
    return items


def reset_backend(self, context):
    backend = Backend.instance()
    backend.reset()

    props = bpy.context.scene.meshgen_props
    props.state = "READY"
    props.history.clear()

    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in ["PREFERENCES", "VIEW_3D"]:
                area.tag_redraw()


class MeshGenPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    __annotations__ = {
        "backend_type": bpy.props.EnumProperty(
            name="Backend type",
            description="Select the backend type to use for generation",
            items=[
                (
                    "LOCAL",
                    "Local",
                    "Use local integrated llama_cpp_python backend for inference",
                ),
                (
                    "REMOTE",
                    "Remote",
                    "Use a remote API for inference (Ollama, OpenAI, etc.)",
                ),
            ],
            default="LOCAL",
            update=reset_backend,
        ),
        "current_model": bpy.props.EnumProperty(
            name="Current model",
            description="Select model for local inference",
            items=get_downloaded_models,
            update=reset_backend,
        ),
        "download_repo_id": bpy.props.StringProperty(
            name="Repository ID",
            description="Hugging Face repository ID for the model",
            default="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        ),
        "download_filename": bpy.props.StringProperty(
            name="Filename",
            description="Filename of the model to download",
            default="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ),
        "llm_provider": bpy.props.EnumProperty(
            name="Provider",
            description="Select the provider to use for remote inference",
            items=[
                (
                    "huggingface",
                    "Hugging Face",
                    "Use Hugging Face for remote inference",
                ),
                ("ollama", "Ollama", "Use Ollama for remote inference"),
                ("anthropic", "Anthropic", "Use Anthropic API for remote inference"),
                ("openai", "OpenAI", "Use OpenAI API for remote inference"),
            ],
            default="huggingface",
            update=reset_backend,
        ),
        "huggingface_model_id": bpy.props.StringProperty(
            name="Model ID",
            description="ID of the model to use",
            default="meta-llama/Llama-3.3-70B-Instruct",
            update=reset_backend,
        ),
        "huggingface_api_key": bpy.props.StringProperty(
            name="API key",
            description="Hugging Face API key",
            default="",
            subtype="PASSWORD",
            update=reset_backend,
        ),
        "ollama_endpoint": bpy.props.StringProperty(
            name="Endpoint",
            description="Base URL for the Ollama server",
            default="http://localhost:11434",
            update=reset_backend,
        ),
        "ollama_model_name": bpy.props.StringProperty(
            name="Model name",
            description="Name of the model to use",
            default="gemma3",
            update=reset_backend,
        ),
        "ollama_api_key": bpy.props.StringProperty(
            name="API key (optional)",
            description="Ollama API key (optional, used for private models)",
            default="",
            subtype="PASSWORD",
            update=reset_backend,
        ),
        "anthropic_model_id": bpy.props.StringProperty(
            name="Model ID",
            description="ID of the model to use",
            default="claude-3-5-sonnet-latest",
            update=reset_backend,
        ),
        "anthropic_api_key": bpy.props.StringProperty(
            name="API key",
            description="Anthropic API key",
            default="",
            subtype="PASSWORD",
            update=reset_backend,
        ),
        "openai_model_id": bpy.props.StringProperty(
            name="Model ID",
            description="ID of the model to use",
            default="gpt-4o",
            update=reset_backend,
        ),
        "openai_api_key": bpy.props.StringProperty(
            name="API key",
            description="OpenAI API key",
            default="",
            subtype="PASSWORD",
            update=reset_backend,
        ),
        "downloading": bpy.props.BoolProperty(default=False),
        "download_progress": bpy.props.FloatProperty(
            subtype="PERCENTAGE",
            min=0.0,
            max=100.0,
            precision=0,
        ),
        "show_generation_settings": bpy.props.BoolProperty(
            name="Show Generation Settings",
            description="Show or hide generation settings",
            default=False,
        ),
        "temperature": bpy.props.FloatProperty(
            name="Temperature",
            description="Controls randomness in generation (higher = more creative, lower = more predictable)",
            default=0.7,
            min=0.0,
            max=1.0,
        ),
        "context_length": bpy.props.IntProperty(
            name="Context Length",
            description="Controls the maximum number of tokens in the context window",
            default=32768,
            min=1024,
            max=65536,
        ),
        "show_integrations_settings": bpy.props.BoolProperty(
            name="Show Integrations Settings",
            description="Show or hide integrations settings",
            default=False,
        ),
        "enable_hyper3d": bpy.props.BoolProperty(
            name="Enable Hyper3D",
            description="Enable Hyper3D for mesh generation",
            default=False,
        ),
        "hyper3d_api_key": bpy.props.StringProperty(
            name="API key",
            description="Hyper3D API key",
            default="awesomemcp",
            subtype="PASSWORD",
        ),
    }

    def draw(self, context):
        layout = self.layout

        backend_box = layout.box()
        backend_box.label(text="Backend", icon="SETTINGS")

        info_box = backend_box.box()
        info_box.label(
            text="Local backend requires at least 8GB VRAM to run on GPU.",
            icon="INFO",
        )
        info_box.label(
            text="For lower-end machines and to run larger models, Remote backend is recommended.",
        )

        backend_box.prop(self, "backend_type", expand=True)

        if self.backend_type == "LOCAL":
            local_box = layout.box()

            header = local_box.row(align=True)
            header.label(text="Downloaded models", icon="PACKAGE")
            header.operator("meshgen.open_models_folder", text="", icon="FILE_FOLDER")

            models = get_available_models()
            if models:
                local_box.prop(self, "current_model")
            elif not self.downloading:
                local_box.label(text="No models downloaded.", icon="INFO")

                op = local_box.operator(
                    "meshgen.download_model",
                    text="Download Recommended Model",
                    icon="IMPORT",
                )
                op.repo_id = self.download_repo_id
                op.filename = self.download_filename

            if self.downloading:
                row = local_box.row(align=True)
                row.label(text="Downloading...")
                row.prop(self, "download_progress", slider=True, text="")

        else:
            remote_box = layout.box()

            remote_box.prop(self, "llm_provider")
            remote_box.separator()

            if self.llm_provider == "ollama":
                remote_box.prop(self, "ollama_endpoint")
                remote_box.prop(self, "ollama_model_name")
                remote_box.prop(self, "ollama_api_key")

            elif self.llm_provider == "huggingface":
                remote_box.prop(self, "huggingface_model_id")
                remote_box.prop(self, "huggingface_api_key")

            elif self.llm_provider == "anthropic":
                remote_box.prop(self, "anthropic_model_id")
                remote_box.prop(self, "anthropic_api_key")

            elif self.llm_provider == "openai":
                remote_box.prop(self, "openai_model_id")
                remote_box.prop(self, "openai_api_key")

        options_box = layout.box()
        header = options_box.row(align=True)
        header.prop(
            self,
            "show_generation_settings",
            icon="TRIA_DOWN" if self.show_generation_settings else "TRIA_RIGHT",
            icon_only=True,
            emboss=False,
        )
        header.label(text="Generation Settings")

        if self.show_generation_settings:
            options_box.prop(self, "temperature", slider=True)
            options_box.prop(self, "context_length", slider=True)

        plugin_box = layout.box()
        header = plugin_box.row(align=True)
        header.prop(
            self,
            "show_integrations_settings",
            icon="TRIA_DOWN" if self.show_integrations_settings else "TRIA_RIGHT",
            icon_only=True,
            emboss=False,
        )
        header.label(text="Integrations")

        if self.show_integrations_settings:
            llama_mesh_box = plugin_box.box()

            llama_mesh_box.label(text="LLaMA-Mesh", icon="PACKAGE")

            llama_mesh_box.label(
                text="Use LLaMA-Mesh for local mesh generation and understanding.",
            )

            if LlamaMeshModelManager.instance().is_loaded:
                llama_mesh_box.label(text="LLaMA-Mesh is loaded", icon="CHECKBOX_HLT")
                llama_mesh_box.operator(
                    "meshgen.unload_llama_mesh", text="Unload LLaMA-Mesh", icon="X"
                )
            else:
                llama_mesh_box.label(
                    text="Requires 5GB VRAM. Not recommended when using local backend.",
                    icon="ERROR",
                )
                llama_mesh_box.operator(
                    "meshgen.load_llama_mesh", text="Load LLaMA-Mesh", icon="IMPORT"
                )

            hyper3d_box = plugin_box.box()
            hyper3d_box.label(text="Hyper3D", icon="PACKAGE")

            hyper3d_box.label(
                text="Use Hyper3D (Rodin) API for mesh generation.",
            )

            hyper3d_box.prop(self, "enable_hyper3d")
            if self.enable_hyper3d:
                hyper3d_box.prop(self, "hyper3d_api_key")


def register():
    bpy.utils.register_class(MeshGenPreferences)


def unregister():
    bpy.utils.unregister_class(MeshGenPreferences)
