import bpy
import os
import sys
import traceback

from ..operators.install_dependencies import load_dependencies
from ..utils import absolute_path

class Generator:
    _instance = None
    

    def __new__(cls):
        if not cls._instance:
            if not cls._instance:
                cls._instance = super(Generator, cls).__new__(cls)
                cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
        import json
        manifest_path = absolute_path(".models/manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.required_models = manifest["required_models"]
        self.downloaded_models = []
        self.dependencies_installed = False
        self.dependencies_loaded = False
        self.llm = None
        self.initialized = True

    def _list_downloaded_models(self):
        models = []

        models_dir = absolute_path(".models")

        if not os.path.exists(models_dir):
            self.downloaded_models = models
            return

        for filename in os.listdir(models_dir):
            if filename.endswith(".gguf"):
                models.append(filename)

        self.downloaded_models = models
    
    def _ensure_dependencies(self):
        self.dependencies_installed = len(os.listdir(absolute_path(".python_dependencies"))) > 2

        if self.dependencies_installed and not self.dependencies_loaded:
            load_dependencies()
            self.dependencies_loaded = True

    def has_dependencies(self):
        self._ensure_dependencies()
        return self.dependencies_installed
    
    def has_required_models(self):
        self._list_downloaded_models()
        return all(model["filename"] in self.downloaded_models for model in self.required_models)
    
    def is_generator_loaded(self):
        return self.llm is not None

    def load_generator(self):
        print("Loading generator...")

        self._ensure_dependencies()

        try:
            if bpy.context.scene.meshgen_props.use_ollama_backend:
                from ollama import Client
                self.llm = Client(
                    host=bpy.context.scene.meshgen_props.ollama_host,
                )
                self.llm.pull(model='hf.co/bartowski/LLaMA-Mesh-GGUF:Q4_K_M')
            else:
                import llama_cpp

                self.llm = llama_cpp.Llama(
                    model_path=absolute_path(".models/LLaMA-Mesh-Q4_K_M.gguf"),
                    n_gpu_layers=-1,
                    seed=1337,
                    n_ctx=4096,
                )

            print("Finished loading generator.")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Failed to load generator: Insufficient memory.", file=sys.stderr)
            else:
                print(f"Failed to load generator: {e}", file=sys.stderr)
            traceback.print_exc()

        except Exception as e:
            print(f"Failed to load generator: {e}", file=sys.stderr)
            traceback.print_exc()

    @classmethod
    def instance(cls):
        return cls()
