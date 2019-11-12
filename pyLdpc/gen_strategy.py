import enum
import pyLdpc_internal

class GEN_STRATEGY(enum.IntEnum):
    First = pyLdpc_internal.GEN_STRATEGY_FIRST
    Mincol = pyLdpc_internal.GEN_STRATEGY_MINCOL
    Minprod = pyLdpc_internal.GEN_STRATEGY_MINPROD