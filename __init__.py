import bpy

from .operators import *
from .panels import *
from .preferences import *
from .property_groups import *

classes = (
    MeshGenPreferences,
    MeshGenProperties,
    MESHGEN_OT_InstallDependencies,
    MESHGEN_OT_UninstallDependencies,
    MESHGEN_OT_DownloadRequiredModels,
    MESHGEN_OT_LoadGenerator,
    MESHGEN_OT_GenerateMesh,
    MESHGEN_OT_CancelGeneration,
    MESHGEN_PT_Panel,
    MESHGEN_PT_Settings,
    MESHGEN_PT_Warning,
    MESHGEN_PT_Setup,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.meshgen_props = bpy.props.PointerProperty(type=MeshGenProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.meshgen_props
