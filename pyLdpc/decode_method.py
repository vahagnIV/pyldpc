import enum
import pyLdpc_internal

class DECODE_METHOD(enum.IntEnum):
    Enum_bit = pyLdpc_internal.DECODE_METHOD_Enum_bit
    Enum_block = pyLdpc_internal.DECODE_METHOD_Enum_block
    Prprp = pyLdpc_internal.DECODE_METHOD_Prprp