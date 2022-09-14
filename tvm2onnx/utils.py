import os
import typing


def print_path_contents(dir_root):
    for path in get_path_contents(dir_root):
        print(path)


def get_path_contents(dir_root) -> typing.List[str]:
    dir_root = str(dir_root)
    contents = []
    for lists in os.listdir(dir_root):
        path = os.path.join(dir_root, lists)
        # Add 1 to catch the trailing / character
        contents.append(path[len(dir_root) + 1 :])
        if os.path.isdir(path):
            contents.extend(get_path_contents(path))
    return contents


def gen_static_library_name(base_name: str) -> str:
    from sys import platform

    if platform == "linux" or platform == "linux2":
        return f"lib{base_name}.a"
    elif platform == "darwin":
        raise Exception("MacOS not supported")
    elif platform == "win32":
        return f"{base_name}.lib"


def gen_shared_library_name(base_name: str) -> str:
    from sys import platform

    if platform == "linux" or platform == "linux2":
        return f"lib{base_name}.so"
    elif platform == "darwin":
        raise Exception("MacOS not supported")
    elif platform == "win32":
        return f"{base_name}.dll"
