import json
import os
import sys
from abc import ABC, abstractmethod

import bpy

from .utils import absolute_path


class LocalModelManager:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LocalModelManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
        manifest_path = absolute_path("models/manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.default_model = manifest["default_model"]
        self.downloaded_models = []
        self.initialized = True
        self.update_downloaded_models()

    def update_downloaded_models(self):
        models = []
        models_dir = absolute_path("models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".gguf"):
                    models.append(file)
        self.downloaded_models = models

    def has_default_model(self):
        self.update_downloaded_models()
        return self.default_model["filename"] in self.downloaded_models

    def get_model_path(self, filename):
        return absolute_path(f"models/{filename}")

    @classmethod
    def instance(cls):
        return cls()


class LLMBackend(ABC):
    @abstractmethod
    def is_valid(self):
        pass

    @abstractmethod
    def is_loaded(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def create_chat_completion(self, messages, stream=True, temperature=0.9):
        pass

    def parse_chunk(self, chunk):
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content"):
                return delta.content
        elif (
            isinstance(chunk, dict) and "choices" in chunk and len(chunk["choices"]) > 0
        ):
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content")
            return content
        return None


class LocalLLM(LLMBackend):
    def __init__(self):
        self.llm = None

    def is_valid(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.current_model is None:
            return False
        model_manager = LocalModelManager.instance()
        model_path = model_manager.get_model_path(prefs.current_model)
        return model_path is not None and os.path.exists(model_path)

    def is_loaded(self):
        return self.llm is not None

    def load(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        model_manager = LocalModelManager.instance()
        model_path = model_manager.get_model_path(prefs.current_model)
        print(f"Loading local model: {model_path}")
        try:
            import llama_cpp

            self.llm = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                seed=1337,
                n_ctx=4096,
            )
            print("Finished loading local model.")
        except Exception as e:
            print(f"Failed to load local model: {e}", file=sys.stderr)
            raise

    def create_chat_completion(self, messages, stream=True, temperature=0.9):
        if self.llm is None:
            raise RuntimeError("Model not loaded")
        return self.llm.create_chat_completion(
            messages, stream=stream, temperature=temperature
        )


class LiteLLMBackend(LLMBackend):
    def __init__(self):
        self.client = None

    def is_valid(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        return (
            prefs.backend_type == "REMOTE"
            and (
                prefs.litellm_provider == "ollama"
                and prefs.ollama_endpoint
                and prefs.ollama_model_name
            )
            or (
                prefs.litellm_provider == "huggingface"
                and prefs.huggingface_model_name
            )
        )

    def is_loaded(self):
        return self.client is not None

    def load(self):
        prefs = bpy.context.preferences.addons[__package__].preferences
        api_base = None
        api_key = None
        model_name = None

        if prefs.litellm_provider == "ollama":
            api_base = prefs.ollama_endpoint
            model_name = prefs.ollama_model_name
        elif prefs.litellm_provider == "huggingface":
            api_key = prefs.huggingface_api_key
            api_base = prefs.huggingface_endpoint
            model_name = prefs.huggingface_model_name

        api_key = api_key if api_key else None
        api_base = api_base if api_base else None

        print(f"Loading LiteLLM with model: {model_name}")
        try:
            import litellm

            if api_key:
                litellm.api_key = api_key

            if api_base:
                litellm.api_base = api_base

            self.client = True
            print("LiteLLM loaded successfully")
        except Exception as e:
            print(f"Failed to load LiteLLM: {e}", file=sys.stderr)
            raise

    def create_chat_completion(self, messages, stream=True, temperature=0.9):
        if not self.is_loaded():
            raise RuntimeError("LiteLLM not loaded")

        try:
            import litellm

            prefs = bpy.context.preferences.addons[__package__].preferences
            provider = prefs.litellm_provider
            model_name = prefs.litellm_model_name

            model = f"{provider}/{model_name}"

            kwargs = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
            }

            if provider == "ollama":
                system_msgs = [msg for msg in messages if msg["role"] == "system"]
                if system_msgs:
                    kwargs["messages"] = [
                        msg for msg in messages if msg["role"] != "system"
                    ]
                    kwargs["system"] = system_msgs[0]["content"]

            return litellm.completion(**kwargs)
        except Exception as e:
            print(f"Error in LiteLLM completion: {e}", file=sys.stderr)
            raise


class MESHGEN_OT_DownloadModel(bpy.types.Operator):
    bl_idname = "meshgen.download_model"
    bl_label = "Download Model"
    bl_description = "Download the recommended model from Hugging Face"
    bl_options = {"REGISTER", "INTERNAL"}

    repo_id: bpy.props.StringProperty(name="Repo ID")
    filename: bpy.props.StringProperty(name="Filename")

    _timer = None
    _download_thread = None
    _progress_queue = None

    def execute(self, context):
        import queue
        import re
        import sys
        import threading

        from huggingface_hub import hf_hub_download

        self._progress_queue = queue.Queue()

        class TqdmCapture:
            def __init__(self, queue, stream):
                self.queue = queue
                self.stream = stream
                self.original_write = stream.write
                self.original_flush = stream.flush

            def write(self, string):
                match = re.search(r"\r.*?(\d+)%", string)
                if match:
                    try:
                        percentage = int(match.group(1))
                        self.queue.put(percentage)
                    except Exception as e:
                        self.queue.put(f"Error parsing progress: {string}, {e}")
                self.original_write(string)
                self.original_flush()

            def flush(self):
                self.original_flush()

        def download_task():
            try:
                old_stderr = sys.stderr
                sys.stderr = TqdmCapture(self._progress_queue, sys.stderr)
                hf_hub_download(
                    self.repo_id,
                    filename=self.filename,
                    local_dir=absolute_path("models"),
                )
                self._progress_queue.put("finished")
            except Exception as e:
                self._progress_queue.put(f"Error downloading model: {e}")
            finally:
                sys.stderr = old_stderr

        prefs = context.preferences.addons[__package__].preferences
        prefs.downloading = True
        prefs.download_progress = 0

        self._download_thread = threading.Thread(target=download_task)
        self._download_thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            prefs = context.preferences.addons[__package__].preferences
            model_manager = LocalModelManager.instance()
            while not self._progress_queue.empty():
                item = self._progress_queue.get()
                if isinstance(item, (int, float)):
                    prefs.download_progress = item
                elif isinstance(item, str):
                    if item == "finished":
                        prefs.downloading = False
                        model_manager.update_downloaded_models()
                        for area in context.screen.areas:
                            if area.type == "PREFERENCES" or area.type == "VIEW_3D":
                                area.tag_redraw()
                        self.cleanup(context)
                        return {"FINISHED"}
                    else:
                        self.report({"ERROR"}, item)
                        prefs.downloading = False
                        model_manager.update_downloaded_models()
                        for area in context.screen.areas:
                            if area.type == "PREFERENCES" or area.type == "VIEW_3D":
                                area.tag_redraw()
                        self.cleanup(context)
                        return {"CANCELLED"}

            for area in context.screen.areas:
                if area.type == "PREFERENCES":
                    area.tag_redraw()

            return {"RUNNING_MODAL"}
        else:
            return {"PASS_THROUGH"}

    def cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        return {"CANCELLED"}


class MESHGEN_OT_SelectModel(bpy.types.Operator):
    bl_idname = "meshgen.select_model"
    bl_label = "Select Model"
    bl_description = "Select the model to use for generation"
    bl_options = {"REGISTER", "INTERNAL"}

    model: bpy.props.StringProperty(name="Model")

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        prefs.current_model = self.model
        return {"FINISHED"}


class MESHGEN_OT_OpenModelsFolder(bpy.types.Operator):
    bl_idname = "meshgen.open_models_folder"
    bl_label = "Open Models Folder"
    bl_description = "Open the models directory in the file explorer"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        models_dir = absolute_path("models")

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            if os.name == "nt":
                os.startfile(models_dir)
            elif os.name == "posix":
                import subprocess

                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.run([opener, models_dir])
        except Exception as e:
            self.report({"ERROR"}, f"Failed to open models folder: {e}")
            return {"CANCELLED"}

        return {"FINISHED"}


class MESHGEN_OT_RefreshModels(bpy.types.Operator):
    bl_idname = "meshgen.refresh_models"
    bl_label = "Refresh Models"
    bl_description = "Refresh the list of downloaded models"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        model_manager = LocalModelManager.instance()
        model_manager.update_downloaded_models()
        for area in bpy.context.screen.areas:
            if area.type == "PREFERENCES" or area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


def register():
    bpy.utils.register_class(MESHGEN_OT_DownloadModel)
    bpy.utils.register_class(MESHGEN_OT_SelectModel)
    bpy.utils.register_class(MESHGEN_OT_OpenModelsFolder)
    bpy.utils.register_class(MESHGEN_OT_RefreshModels)


def unregister():
    bpy.utils.unregister_class(MESHGEN_OT_DownloadModel)
    bpy.utils.unregister_class(MESHGEN_OT_SelectModel)
    bpy.utils.unregister_class(MESHGEN_OT_OpenModelsFolder)
    bpy.utils.unregister_class(MESHGEN_OT_RefreshModels)
