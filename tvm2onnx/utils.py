# Copyright 2023 OctoML
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import typing


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
