import torch
from torch.nn import Softmax
import numpy as np
import matplotlib.pyplot as plt
import random

from constants import *

def get_best_device():
    device = 'cpu'
    if torch.has_mps:
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


