import os
import sys

import bpy


def absolute_path(path: str):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), path)


def open_console():
    if not sys.platform == "win32":
        return

    bpy.ops.wm.console_toggle()
