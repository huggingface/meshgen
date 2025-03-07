import os
import sys
import traceback

from .utils import absolute_path


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

        manifest_path = absolute_path("models/manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.required_models = manifest["required_models"]
        self.downloaded_models = []
        self.llm = None
        self.initialized = True

    def _list_downloaded_models(self):
        models = []

        models_dir = absolute_path("models")

        if not os.path.exists(models_dir):
            self.downloaded_models = models
            return

        for filename in os.listdir(models_dir):
            if filename.endswith(".gguf"):
                models.append(filename)

        self.downloaded_models = models

    def has_required_models(self):
        self._list_downloaded_models()
        return all(
            model["filename"] in self.downloaded_models
            for model in self.required_models
        )

    def is_generator_loaded(self):
        return self.llm is not None

    def load_generator(self):
        print("Loading generator...")

        try:
            import llama_cpp

            self.llm = llama_cpp.Llama(
                model_path=absolute_path("models/LLaMA-Mesh-Q4_K_M.gguf"),
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
