import pyLdpc_internal
from .generator_matrix import GeneratorMatrix
from .pchk_matrix import ParityCheckMatrix
from .gen_method import GEN_METHOD
from .gen_strategy import GEN_STRATEGY


class GeneratorMatrixMaker:
    def __init__(self, method: int = GEN_METHOD.Dense, strategy: int = GEN_STRATEGY.First, abandon_number: int = 0,
                 abandon_when: int = 0):
        self.method = method
        self.strategy = strategy
        self.abandon_number = abandon_number
        self.abandon_when = abandon_when
        pass

    def generate(self, H: ParityCheckMatrix):
        print(dir(H))
        gen_matrix = pyLdpc_internal.make_gen(method=self.method, H=getattr(H, '__internal_matrix'),
                                                   strategy=self.strategy,
                                                   abandon_number=self.abandon_number,
                                                   abandon_when=self.abandon_when)
        result = GeneratorMatrix()
        setattr(result, '__internal_matrix', gen_matrix)
        return result
