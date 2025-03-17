import importlib

from . import backend, operators, preferences, properties, tools, ui, utils

if "bpy" in locals():
    importlib.reload(backend)
    importlib.reload(operators)
    importlib.reload(preferences)
    importlib.reload(properties)
    importlib.reload(tools)
    importlib.reload(ui)
    importlib.reload(utils)


def reset_backend():
    try:
        backend.Backend.reset()
    except Exception:
        pass


def reset_runtime_preferences():
    try:
        import bpy

        if __package__ in bpy.context.preferences.addons:
            prefs = bpy.context.preferences.addons[__package__].preferences
            prefs.downloading = False
            prefs.download_progress = 0
    except Exception:
        pass


def register():
    operators.register()
    ui.register()
    preferences.register()
    properties.register()

    print(f"{__package__} is registered")


def unregister():
    reset_backend()
    reset_runtime_preferences()

    operators.unregister()
    ui.unregister()
    preferences.unregister()
    properties.unregister()

    print(f"{__package__} is unregistered")


if __name__ == "__main__":
    register()
