import os

import structlog
import typing

from tvm2onnx.error import TVM2ONNXError

LOG = structlog.get_logger(__name__)


class UnableToSetTvmNumThreadsError(TVM2ONNXError):
    """Indicates that set_tvm_num_threads was unable to change the number of threads."""


def set_tvm_num_threads(tvm_num_threads: int):
    """Sets the number of threads that TVM will use for inference

    This can always be reduced in the current process, but cannot always be
    increased.

    :param tvm_num_threads: the number of threads TVM should use for inference
    :raises: UnableToSetTvmNumThreadsError if the value could not be set
    :raises: ValueError if the value is not a positive integer
    """
    if tvm_num_threads <= 0:
        raise ValueError("set_tvm_num_threads requires a positive integer")

    # Set the environment variable to override MaxConcurrency.
    os.environ["TVM_NUM_THREADS"] = str(tvm_num_threads)

    from tvm._ffi import get_global_func

    config_threadpool = get_global_func("runtime.config_threadpool")
    get_num_threads = get_global_func("runtime.NumThreads")

    if get_num_threads() != tvm_num_threads:
        AFFINITY_MODE_BIG = 1  # kBig (1) is the default AffinityMode
        config_threadpool(AFFINITY_MODE_BIG, tvm_num_threads)

        if get_num_threads() != tvm_num_threads:
            raise UnableToSetTvmNumThreadsError(
                "The number of TVM threads could not be changed; perhaps the "
                "thread pool was already initialized with a smaller number of "
                "threads?\n\n"
                f"{get_num_threads()} != {tvm_num_threads}"
            )


def get_tvm_revision_hash() -> str:
    return os.environ["TVM_REVISION_HASH"]


def print_path_contents(dir_root):
    for path in get_path_contents(dir_root):
        print(path)


def get_path_contents(dir_root) ->typing.List[str]:
    contents = []
    for lists in os.listdir(dir_root):
        path = os.path.join(dir_root, lists)
        contents.append(path)
        if os.path.isdir(path):
            contents.extend(get_path_contents(path))
    return contents
