import bpy

from .backend import Backend

EVENT_LABELS = {
    "THINKING": "Thinking...",
    "STEP": "Step",
    "ERROR": "Error",
    "CANCELED": "Canceled",
    "SUCCESS": "Finished",
    "LOADING": "Loading...",
    "LOADING_SUCCESS": "Loaded model",
    "LOADING_ERROR": "Failed to load model",
}

EVENT_ICONS = {
    "THINKING": "SORTTIME",
    "STEP": "LIGHT",
    "ERROR": "ERROR",
    "CANCELED": "X",
    "SUCCESS": "CHECKMARK",
    "LOADING": "SORTTIME",
    "LOADING_SUCCESS": "CHECKMARK",
    "LOADING_ERROR": "ERROR",
}


class MESHGEN_PT_Panel(bpy.types.Panel):
    bl_label = "Chat"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MeshGen"

    def draw(self, context):
        layout = self.layout
        props = context.scene.meshgen_props

        backend = Backend.instance()
        if not backend.is_valid():
            setup_box = layout.box()
            setup_box.label(text="Finish setup in preferences.", icon="INFO")

            preferences_row = layout.row()
            preferences_row.scale_y = 1.2
            preferences_row.operator(
                "preferences.addon_show", text="Open Preferences", icon="SETTINGS"
            ).module = __package__

            return

        # User
        user_box = layout.box()
        user_box.label(text="You", icon="USER")

        if props.state == "READY":
            user_box.prop(props, "prompt", text="")
        else:
            user_box.label(text=props.prompt)

        action_row = user_box.row(align=True)
        action_row.scale_y = 1.2
        main_button_col = action_row.column(align=True)
        if props.state == "READY":
            main_button_col.operator("meshgen.chat", text="Submit", icon="PLAY")
        elif props.state == "LOADING" or props.state == "RUNNING":
            main_button_col.operator("meshgen.cancel_chat", text="Cancel", icon="X")
        log_button_col = action_row.column(align=True)
        log_button_col.operator("meshgen.open_log", text="", icon="TEXT")

        # Agent
        if not props.history:
            return

        layout.separator()
        agent_box = layout.box()
        agent_box.label(text="Agent", icon="LIGHT")

        for event in props.history:
            event_box = agent_box.box()
            event_box.label(text=EVENT_LABELS[event.type], icon=EVENT_ICONS[event.type])
            if event.content:
                event_box.label(text=event.content)


def register():
    bpy.utils.register_class(MESHGEN_PT_Panel)


def unregister():
    bpy.utils.unregister_class(MESHGEN_PT_Panel)
