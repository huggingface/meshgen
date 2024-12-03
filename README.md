# MeshGen

A Blender addon for generating meshes with AI.

![meshgen](docs/meshgen.gif)

This initial release contains a minimal integration of [LLaMa-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh) in Blender.

# Installation

Go to the [Latest Release](https://github.com/huggingface/meshgen/releases/latest) page for a download link and installation instructions.

# Setup
- In `MeshGen` addon preferences, click `Download Required Models`. If the automatic download fails, download the models directly from [HuggingFace repo](https://huggingface.co/Zhengyi/LLaMA-Mesh).
- If set up correctly, addon preferences should display the message `Ready to generate. Press 'N' -> MeshGen to get started.'

# Usage

- Press `N` -> `MeshGen` (or `View` -> `Sidebar` -> Select the `MeshGen` tab)
- Click `Load Generator` (this will take a while)
- Enter a prompt, for example: `Create a 3D obj file using the following description: a desk`
- Click `Generate Mesh`

# Troubleshooting

- ModuleNotFoundError when loading generator:
  - Go to Scripting and give the following command in the Python Interactive Console to get the path to your modules folder
    ```
    >>> bpy.utils.user_resource("SCRIPTS", path="modules")
    ```
  - Install missing modules by specifying target using pip. For example installing PyTorch in Windows with cuda 12.1:
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --target=<path-to-your-modules-folder>
    ```
- Find errors in the console:
  - Windows: In Blender, go to `Window` -> `Toggle System Console`
  - Mac/Linux: Launch Blender from the terminal
- Report errors in [Issues](https://github.com/huggingface/meshgen/issues)
