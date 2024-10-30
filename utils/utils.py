import random
import numpy as np
import torch


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def pretty_print_matrix(m, name='Mat', num=4):
    if isinstance(m, torch.Tensor):
        matrix = m.detach().cpu().numpy()
    else:
        matrix = m
    print('\u0332'.join(name + ' '))
    print('\n'.join(['\t'.join([str(round(cell.item(),num)) for cell in row]) for row in matrix]))    