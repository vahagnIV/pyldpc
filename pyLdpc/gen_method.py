import enum
import pyLdpc_internal

class GEN_METHOD(enum.IntEnum):
    Mixed = pyLdpc_internal.GEN_METHOD_MIXED
    Sparse = pyLdpc_internal.GEN_METHOD_SPARSE
    Dense = pyLdpc_internal.GEN_METHOD_DENSE