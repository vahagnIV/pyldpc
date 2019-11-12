from pyLdpc import ParityCheckMatrix, GeneratorMatrix
import numpy as np
import pyLdpc_internal


class Encoder:
    def __init__(self, H: ParityCheckMatrix, G: GeneratorMatrix):
        self.H = H
        self.G = G

    def encode(self, data: np.ndarray) -> np.ndarray:
        return pyLdpc_internal.encode(H=getattr(self.H, '__internal_matrix'), G=getattr(self.G, '__internal_matrix'),
                               payload=data)
