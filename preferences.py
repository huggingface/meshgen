import bpy

from .backend import LocalModelManager


def update_backend(self, context):
    from .generator import Generator

    generator = Generator.instance()
    generator._backend = None

    for area in bpy.context.screen.areas:
        if area.type in ["PREFERENCES", "VIEW_3D"]:
            area.tag_redraw()


class MeshGenPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    backend_type: bpy.props.EnumProperty(
        name="Backend Type",
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
        update=update_backend,
    )

    current_model: bpy.props.StringProperty(
        name="Current Model",
        description="Select model for local inference",
        default="LLaMA-Mesh-Q4_K_M.gguf",
        update=update_backend,
    )

    litellm_provider: bpy.props.EnumProperty(
        name="Provider",
        description="Select the provider to use for remote inference",
        items=[
            ("huggingface", "Hugging Face", "Use Hugging Face for remote inference"),
            ("ollama", "Ollama", "Use Ollama for remote inference"),
        ],
        default="huggingface",
        update=update_backend,
    )

    huggingface_api_key: bpy.props.StringProperty(
        name="API Key (optional)",
        description="Hugging Face API key (optional, only if not set in environment variables)",
        default="",
        subtype="PASSWORD",
        update=update_backend,
    )

    huggingface_endpoint: bpy.props.StringProperty(
        name="Endpoint (optional)",
        description="Endpoint URL (optional, only for inference endpoints)",
        default="",
        update=update_backend,
    )

    huggingface_model_name: bpy.props.StringProperty(
        name="Model Name",
        description="Name of the model to use",
        default="Zhengyi/LLaMA-Mesh",
        update=update_backend,
    )

    ollama_endpoint: bpy.props.StringProperty(
        name="Endpoint",
        description="Base URL for the Ollama server",
        default="http://localhost:11434",
        update=update_backend,
    )

    ollama_model_name: bpy.props.StringProperty(
        name="Model Name",
        description="Name of the model to use",
        default="hf.co/bartowski/LLaMA-Mesh-GGUF:Q4_K_M",
        update=update_backend,
    )

    downloading: bpy.props.BoolProperty(default=False)
    download_progress: bpy.props.FloatProperty(
        subtype="PERCENTAGE",
        min=0.0,
        max=100.0,
        precision=0,
    )

    def draw(self, context):
        layout = self.layout

        backend_box = layout.box()
        backend_box.label(text="Backend Selection", icon="SETTINGS")
        backend_box.prop(self, "backend_type", expand=True)

        layout.separator()

        if self.backend_type == "LOCAL":
            model_manager = LocalModelManager.instance()

            local_box = layout.box()

            if self.downloading:
                row = local_box.row(align=True)
                row.label(text="Downloading...")
                row.prop(self, "download_progress", slider=True, text="")
            else:
                header = local_box.row(align=True)
                header.label(text="Installed Models", icon="PACKAGE")
                header.operator(
                    "meshgen.open_models_folder", text="", icon="FILE_FOLDER"
                )
                header.operator("meshgen.refresh_models", text="", icon="FILE_REFRESH")

                if not model_manager.downloaded_models and not self.downloading:
                    local_box.label(text="No models downloaded.", icon="INFO")

                    op = local_box.operator(
                        "meshgen.download_model",
                        text="Download Recommended Model",
                        icon="IMPORT",
                    )
                    op.repo_id = model_manager.default_model["repo_id"]
                    op.filename = model_manager.default_model["filename"]
                else:
                    for model in model_manager.downloaded_models:
                        row = local_box.row(align=True)
                        is_active = model == self.current_model
                        row.active = is_active
                        op = row.operator(
                            "meshgen.select_model",
                            text=model,
                            emboss=False,
                            depress=is_active,
                        )
                        op.model = model

                    if self.current_model:
                        local_box.label(
                            text=f"Currently using {self.current_model}", icon="INFO"
                        )
                    else:
                        local_box.label(text="No model selected", icon="WARNING")

        else:
            remote_box = layout.box()

            remote_box.prop(self, "litellm_provider")
            remote_box.separator()

            if self.litellm_provider == "ollama":
                remote_box.prop(self, "ollama_endpoint")
                remote_box.prop(self, "ollama_model_name")

            elif self.litellm_provider == "huggingface":
                remote_box.prop(self, "huggingface_api_key")
                remote_box.prop(self, "huggingface_endpoint")
                remote_box.prop(self, "huggingface_model_name")


def register():
    bpy.utils.register_class(MeshGenPreferences)


def unregister():
    bpy.utils.unregister_class(MeshGenPreferences)
