# MeshGen

A Blender addon for generating meshes with AI.

![meshgen](docs/meshgen.gif)

This initial release contains a minimal integration of [LLaMA-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh) in Blender.

# Installation

Go to the [Latest Release](https://github.com/huggingface/meshgen/releases/latest) page for a download link and installation instructions.

# Usage

- Press `N` -> `MeshGen` (or `View` -> `Sidebar` -> Select the `MeshGen` tab)
- Click `Load Generator` (this will take a while)
- Enter a prompt, for example: `Create a 3D obj file using the following description: a desk`
- Click `Generate Mesh`

# Troubleshooting

- Find errors in the console:
  - Windows: In Blender, go to `Window` -> `Toggle System Console`
  - Mac/Linux: Launch Blender from the terminal
- Report errors in [Issues](https://github.com/huggingface/meshgen/issues)
