import numpy as np

from tvm2onnx.error import PackagingError


def tvm_load_late_bound_constants(self, consts_path):
    """This function is horrific. This is the only way I can get constants from
    tvm. All I can say is that it works."""
    # https://docs.python.org/3/library/struct.html
    import ctypes
    import struct

    kTVMNDArrayListMagic = int(hex(0xF7E58D4F05049CB7), 16)
    kTVMNDArrayMagic = int(hex(0xDD5E40F096B4A13F), 16)

    class DLDataType(ctypes.Structure):
        TYPE_MAP = {
            (1, 1, 1): "bool",
            (0, 32, 1): "int32",
            (0, 64, 1): "int64",
            (1, 32, 1): "uint32",
            (1, 64, 1): "uint64",
            (2, 16, 1): "float16",
            (2, 32, 1): "float32",
            (2, 64, 1): "float64",
        }

    constants = {}
    names = []
    with open(consts_path, "rb") as f:
        magic = struct.unpack("Q", f.read(8))[0]
        if magic != kTVMNDArrayListMagic:
            raise PackagingError("No magic in consts file")
        reserved = struct.unpack("Q", f.read(8))[0]
        # std::vector<std::string> names;
        # ICHECK(strm->Read(&names)) << "Invalid parameters file format";
        name_count = struct.unpack("Q", f.read(8))[0]
        for i in range(name_count):
            name_length = struct.unpack("Q", f.read(8))[0]
            name = f.read(name_length).decode("utf-8")
            names.append(name)
        data_count = struct.unpack("Q", f.read(8))[0]
        for i in range(data_count):
            magic = struct.unpack("Q", f.read(8))[0]
            if magic != kTVMNDArrayMagic:
                raise PackagingError("Data not array")
            reserved = struct.unpack("Q", f.read(8))[0]
            f.read(reserved)  # skip reserved space
            # DLDevice device;
            device_type = struct.unpack("I", f.read(4))[0]  # noqa
            device_id = struct.unpack("I", f.read(4))[0]  # noqa
            # int ndim
            ndim = struct.unpack("I", f.read(4))[0]
            # DLDataType dtype;
            dtype_code = struct.unpack("B", f.read(1))[0]
            dtype_bits = struct.unpack("B", f.read(1))[0]
            dtype_lanes = struct.unpack("H", f.read(2))[0]
            shape = []
            for dim in range(ndim):
                axis = struct.unpack("Q", f.read(8))[0]
                shape.append(axis)
            data_byte_size = struct.unpack("Q", f.read(8))[0]
            data = f.read(data_byte_size)
            dtype_str = (dtype_code, dtype_bits, dtype_lanes)
            dtype = DLDataType.TYPE_MAP[dtype_str]
            array = np.ndarray(shape=shape, dtype=dtype, buffer=data)
            constants[names[i]] = array
    return constants
