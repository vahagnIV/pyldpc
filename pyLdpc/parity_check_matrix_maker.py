import pyLdpc_internal
from .pchk_matrix import ParityCheckMatrix
from .pchk_method import PCHK_METHOD


class ParityCheckMatrixMaker:
    def __init__(self, method=PCHK_METHOD.Evenboth, M: int = 14, N: int = 30, distrib='3'):
        self.method = method
        self.M = M
        self.N = N
        self.distrib = distrib

    def generate(self, seed: int):
        internal_matrix = pyLdpc_internal.make_ldpc(method=self.method, seed=seed, M=self.M, N=self.N,
                                                    method_args=self.distrib)
        result = ParityCheckMatrix()
        setattr(result, '__internal_matrix',  internal_matrix)
        return result
