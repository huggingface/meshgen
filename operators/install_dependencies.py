import ensurepip
import os
import site
import subprocess
import requests
import shutil
import sys
import sysconfig
import tarfile

import bpy

from ..utils import absolute_path, open_console


def install_pip():
    try:
        import pip
    except ImportError:
        ensurepip.bootstrap()


def install_devel():
    python_include_dir = sysconfig.get_paths()["include"]
    print(f"Python include dir: {python_include_dir}")

    if os.path.exists(os.path.join(python_include_dir, "Python.h")):
        print("Python headers already installed, skipping")
        return

    if not os.access(python_include_dir, os.W_OK):
        print("Can't access include dir, skipping", file=sys.stderr)
        return

    version_info = sys.version_info
    version_string = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    download_url = f"https://www.python.org/ftp/python/{version_string}/Python-{version_string}.tgz"
    python_devel_tgz_path = absolute_path("python-devel.tgz")
    response = requests.get(download_url)
    with open(python_devel_tgz_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded Python devel tgz to {python_devel_tgz_path}")

    with tarfile.open(python_devel_tgz_path) as tar:

        def is_include_file(member):
            return member.name.startswith(f"Python-{version_string}/Include/")

        include_members = filter(is_include_file, tar.getmembers())

        for member in include_members:
            member.path = member.name.replace(f"Python-{version_string}/Include/", "")
            tar.extract(member, path=python_include_dir)
            print(f"Extracted {member.name} to {python_include_dir}")

    os.remove(python_devel_tgz_path)


def load_dependencies():
    module_name = os.path.basename(os.path.dirname(__file__))

    dependencies_dir = absolute_path(".python_dependencies")
    if dependencies_dir not in sys.path:
        sys.path.append(dependencies_dir)
    site.addsitedir(dependencies_dir)
    dependencies = sys.path.pop(-1)
    sys.path.insert(0, dependencies)

    for module_name in ["distutils", "setuptools", "_distutils_hack"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    if sys.platform == "win32":
        python3_path = os.path.abspath(
            os.path.join(sys.executable, "..\\..\\..\\..\\python3.dll")
        )
        if os.path.exists(python3_path):
            os.add_dll_directory(os.path.dirname(python3_path))

        mkl_bin = os.path.join(
            os.path.dirname(__file__), ".python_dependencies\\Library\\bin"
        )
        if os.path.exists(mkl_bin):
            os.add_dll_directory(mkl_bin)


def check_cuda():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except Exception:
        return False


def install_and_load_dependencies():
    install_pip()
    install_devel()

    dependencies_dir = absolute_path(".python_dependencies")
    if os.path.exists(dependencies_dir) and len(os.listdir(dependencies_dir)) > 2:
        print("Requirements already installed, skipping")
        load_dependencies()
        return

    os.makedirs(dependencies_dir, exist_ok=True)

    if (sys.platform == "win32" or sys.platform == "linux") and check_cuda():
        requirements_file = absolute_path("./requirements/cuda.txt")
    else:
        requirements_file = absolute_path("./requirements/cpu.txt")
    
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file,
            "--upgrade",
            "--no-cache-dir",
            "--target",
            dependencies_dir,
        ],
        check=True,
        cwd=os.path.dirname(__file__),
    )

    load_dependencies()


class MESHGEN_OT_InstallDependencies(bpy.types.Operator):
    bl_idname = "meshgen.install_dependencies"
    bl_label = "Install Dependencies"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        if sys.platform == "win32":
            open_console()

        install_and_load_dependencies()

        return {"FINISHED"}


class MESHGEN_OT_UninstallDependencies(bpy.types.Operator):
    bl_idname = "meshgen.uninstall_dependencies"
    bl_label = "Uninstall Dependencies"
    bl_options = {"REGISTER", "INTERNAL"}

    def execute(self, context):
        dependencies_dir = absolute_path(".python_dependencies")
        print(f"Uninstalling dependencies from {dependencies_dir}")

        for item in os.listdir(dependencies_dir):
            item_path = os.path.join(dependencies_dir, item)
            if item == ".gitignore":
                continue
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            print(f"Uninstalled {item}")

        print("Finished uninstalling dependencies")

        return {"FINISHED"}
