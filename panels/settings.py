import bpy

from ..generator.generator import Generator


class MESHGEN_PT_Settings(bpy.types.Panel):
    bl_idname = "MESHGEN_PT_settings"
    bl_label = "Options"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MeshGen"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        generator = Generator.instance()
        return generator.is_generator_loaded()

    def draw(self, context):
        layout = self.layout
        props = context.scene.meshgen_props

        layout.prop(props, "temperature")
