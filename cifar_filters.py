import torch
import numpy as np
import random
from torchvision import datasets, transforms

random.seed(42)

kwargs = {'num_workers': 4, 'pin_memory': True} \
         if torch.cuda.is_available() else {}

def corner_kernel(factor):
    kernel = np.ones((32, 32))
    limit = int(factor * kernel.shape[0]) * 2
    f = (1 - np.tril(kernel[:limit, :limit]))
    kernel[kernel.shape[0] - limit:limit + kernel.shape[0], :limit] = f
    return torch.tensor(kernel)

def single_square_kernel(start=5, size=10):
    kernel = np.ones((32, 32))
    kernel[start:start+size, start:start+size] = 0.
    return torch.tensor(kernel)

def single_square_kernel_random(start=5, size=10):
    random_pos = random.randint(0, 31 - size)
    kernel = np.ones((32, 32))
    kernel[random_pos:random_pos + size, random_pos:random_pos + size] = 0.
    return torch.tensor(kernel)

def multiple_square_kernel(n_cuts=3, start=5, size=10):
    size_triple = size // n_cuts
    kernel = np.ones((32, 32))

    beg = start
    end = start + size_triple

    for _ in range(n_cuts):

        kernel[beg:end, beg:end] = 0.

        beg = end + 5
        end = beg + size_triple

    return torch.tensor(kernel)

def multiple_square_kernel_random(n_cuts = 2, start=0, size=20, sep = 5):
    size_triple = size // n_cuts
    kernel = np.ones((32, 32))

    beg1 = start + random.randint(1, sep)
    end1 = beg1 + size_triple

    beg2 = start + random.randint(1, sep)
    end2 = beg2 + size_triple

    for _ in range(n_cuts):

        kernel[beg1:end1, beg2:end2] = 0.

        beg1 = end1 + random.randint(1, sep)
        end1 = beg1 + size_triple

        beg2 = end2 + random.randint(1, sep)
        end2 = beg2 + size_triple

    return torch.tensor(kernel)

