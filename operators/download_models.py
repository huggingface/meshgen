import bpy
import os
import sys

from ..generator.generator import Generator
from ..utils import absolute_path, open_console


class MESHGEN_OT_DownloadRequiredModels(bpy.types.Operator):
    bl_idname = "meshgen.download_required_models"
    bl_label = "Download Required Models"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        if sys.platform == "win32":
            open_console()

        from huggingface_hub import hf_hub_download

        generator = Generator.instance()
        models_to_download = [model for model in generator.required_models if model not in generator.downloaded_models]

        if not models_to_download:
            print("All required models are already downloaded.")
            return
        
        models_dir = absolute_path(".models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for model in models_to_download:
            print(f"Downloading model: {model['repo_id']}:{model['filename']}")
            hf_hub_download(model["repo_id"], filename=model["filename"], local_dir=models_dir)
            generator._list_downloaded_models()
        return {"FINISHED"}
