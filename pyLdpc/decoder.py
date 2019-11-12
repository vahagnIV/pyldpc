import pyLdpc_internal
from .channel import CHANNEL
from .decode_method import DECODE_METHOD
import numpy as np
from .parity_check_matrix_maker import ParityCheckMatrix
from .generator_matrix import GeneratorMatrix


class Decoder:
    def __init__(self, H: ParityCheckMatrix, G: GeneratorMatrix, channel: int = CHANNEL.Bsc,
                 decode_method: int = DECODE_METHOD.Enum_bit, param: float = 0.15):
        self.channel = channel
        self.decode_method = decode_method
        self.param = param
        self.H = H
        self.G = G

    def decode(self, encoded_message: np.ndarray) -> np.ndarray:
        return pyLdpc_internal.decode(H=getattr(self.H, '__internal_matrix'), G=getattr(self.G, '__internal_matrix'),
                                      message=encoded_message, channel_type=self.channel,
                                      decode_method=self.decode_method, param=self.param)

    def extract(self, decoded_message: np.ndarray) -> np.ndarray:
        return pyLdpc_internal.extract(H=getattr(self.H, '__internal_matrix'), G=getattr(self.G, '__internal_matrix'),
                                       message=decoded_message)

    def decodeAndExtract(self, encoded_message: np.ndarray):
        return self.extract(self.decode(encoded_message))
