if "bpy" in locals():
    import importlib

    importlib.reload(backend)
    importlib.reload(generator)
    importlib.reload(ui)
    importlib.reload(preferences)
    importlib.reload(properties)
    importlib.reload(utils)
else:
    from . import backend, generator, preferences, properties, ui


def register():
    backend.register()
    generator.register()
    ui.register()
    preferences.register()
    properties.register()

    print(f"{__package__} is registered")


def unregister():
    backend.unregister()
    generator.unregister()
    ui.unregister()
    preferences.unregister()
    properties.unregister()

    print(f"{__package__} is unregistered")


if __name__ == "__main__":
    register()
