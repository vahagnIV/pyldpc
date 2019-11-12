import enum
import pyLdpc_internal

class CHANNEL(enum.IntEnum):
    Bsc = pyLdpc_internal.CHANNEL_BSC
    Awln = pyLdpc_internal.CHANNEL_AWLN
    Awgn = pyLdpc_internal.CHANNEL_AWGN