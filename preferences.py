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


def register():
    bpy.utils.register_class(MeshGenPreferences)


def unregister():
    bpy.utils.unregister_class(MeshGenPreferences)
