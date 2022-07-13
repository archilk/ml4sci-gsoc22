import torch
from torch.nn import Softmax
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from constants import *

def get_best_device():
    if 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.has_mps:
        device = 'mps'
    else:
        device = 'cpu'
    return device


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


