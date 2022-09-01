import os
import typing

import structlog

LOG = structlog.get_logger(__name__)


def print_path_contents(dir_root):
    for path in get_path_contents(dir_root):
        print(path)


def get_path_contents(dir_root) -> typing.List[str]:
    contents = []
    for lists in os.listdir(dir_root):
        path = os.path.join(dir_root, lists)
        # Add 1 to catch the trailing / character
        contents.append(path[len(dir_root) + 1 :])
        if os.path.isdir(path):
            contents.extend(get_path_contents(path))
    return contents
