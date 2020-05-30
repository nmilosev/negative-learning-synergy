import torch
import numpy as np
from torchvision import datasets, transforms

kwargs = {'num_workers': 4, 'pin_memory': True} \
         if torch.cuda.is_available() else {}

def corner_kernel(factor):
    kernel = np.ones((32, 32))
    limit = int(factor * kernel.shape[0]) * 2
    f = (1 - np.tril(kernel[:limit, :limit]))
    kernel[kernel.shape[0] - limit:limit + kernel.shape[0], :limit] = f
    return torch.tensor(kernel)
