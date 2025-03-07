if "bpy" in locals():
    import imp

    imp.reload(generator)
    imp.reload(operators)
    imp.reload(ui)
    imp.reload(preferences)
    imp.reload(properties)
    imp.reload(utils)
else:
    from . import operators
    from . import ui
    from . import preferences
    from . import properties


import bpy


def register():
    operators.register()
    ui.register()
    preferences.register()
    properties.register()

    print(f"{__package__} is registered")


def unregister():
    operators.unregister()
    ui.unregister()
    preferences.unregister()
    properties.unregister()

    print(f"{__package__} is unregistered")


if __name__ == "__main__":
    register()
