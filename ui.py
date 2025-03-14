import bpy

from .generator import Generator


class MESHGEN_PT_Panel(bpy.types.Panel):
    bl_label = "Generate"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MeshGen"

    def draw(self, context):
        layout = self.layout
        props = context.scene.meshgen_props
        generator = Generator.instance()

        if generator.is_backend_valid():
            if props.vertices_generated > 0 or props.faces_generated > 0:
                results_box = layout.box()
                results_col = results_box.column(align=True)
                results_col.label(text="Results", icon="INFO")

                if props.vertices_generated > 0:
                    results_col.label(text=f"Vertices: {props.vertices_generated}")
                if props.faces_generated > 0:
                    results_col.label(text=f"Faces: {props.faces_generated}")

                if not props.is_running:
                    if props.cancelled:
                        results_col.label(text="Cancelled", icon="X")
                    else:
                        results_col.label(text="Complete", icon="CHECKMARK")

                layout.separator()

            if props.is_running:
                layout.label(text="Generation in progress...", icon="SORTTIME")
                layout.operator("meshgen.cancel", text="Cancel", icon="X")
            else:
                settings_box = layout.box()
                settings_box.label(text="Settings", icon="SETTINGS")
                settings_box.prop(props, "prompt")
                settings_box.prop(props, "temperature", slider=True)

                layout.separator()

                generate_row = layout.row()
                generate_row.scale_y = 1.2
                generate_row.operator("meshgen.generate", text="Generate", icon="PLAY")
        else:
            error_box = layout.box()

            error_box.label(text="Invalid configuration", icon="ERROR")
            error_box.operator(
                "preferences.addon_show", text="Open Preferences"
            ).module = __package__


def register():
    bpy.utils.register_class(MESHGEN_PT_Panel)


def unregister():
    bpy.utils.unregister_class(MESHGEN_PT_Panel)
