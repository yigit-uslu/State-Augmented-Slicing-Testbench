from enum import Enum
import random

class Slice(Enum):
    HT = 0 # high-throughput
    LL = 1 # low-latency
    BE = 2 # best-effort
    IA = 3 # inactive

    # @classmethod
    # def __getitem__(self, i):
    #     return Slice.list()[i]

    @classmethod
    def list(cls):
        # return list(map(lambda c: c.value, cls))
        return list(map(lambda c: c, cls))
    
    @classmethod
    def sample(cls, weights, n_samples):
        slices = cls.list()
        sampled_slice = random.choices(slices, weights=weights, k=n_samples)
        return sampled_slice