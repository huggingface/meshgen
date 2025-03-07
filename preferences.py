import bpy

from .generator import Generator
from .operators import MESHGEN_OT_DownloadRequiredModels


class MeshGenPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    @staticmethod
    def register():
        pass

    def draw(self, context):
        layout = self.layout

        generator = Generator.instance()

        if not generator.has_required_models():
            layout.label(text="Required models not downloaded.", icon="ERROR")
            layout.operator(MESHGEN_OT_DownloadRequiredModels.bl_idname, icon="IMPORT")
        else:
            layout.label(text="Ready to generate. Press 'N' -> MeshGen to get started.")

        layout.separator()

        layout.prop(
            context.scene.meshgen_props,
            "show_developer_options",
            text="Show Developer Options",
        )

        if context.scene.meshgen_props.show_developer_options:
            box = layout.box()

            if bpy.app.online_access:
                box.prop(
                    context.scene.meshgen_props,
                    "use_ollama_backend",
                    text="Use Ollama Backend",
                )

                if context.scene.meshgen_props.use_ollama_backend:
                    ollama_options_box = box.box()
                    ollama_options_box.prop(
                        context.scene.meshgen_props, "ollama_host", text="Ollama Host"
                    )


def register():
    bpy.utils.register_class(MeshGenPreferences)


def unregister():
    bpy.utils.unregister_class(MeshGenPreferences)
