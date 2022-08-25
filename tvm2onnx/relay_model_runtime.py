"""Defines a function to execute a Relay model which only requires the TVM runtime."""

# Standard names for RelayVM library and consts files.
RELAY_VM_LIBRARY_NAME = "lib.so"
RELAY_VM_CONSTS_NAME = "consts"

# Constants above this size in Relay models will be saved separately
# from the module library, into the consts file named above.
RELAY_VM_LARGE_CONST_BYTE_LIMIT = 256
