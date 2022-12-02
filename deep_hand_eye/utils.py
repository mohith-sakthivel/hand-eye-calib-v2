from typing import List

import torch
from torch import Tensor
from torch_geometric.utils import degree


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    """
    Finds how many items belong to each graph. Splits the given tensor based on the number of
    item each graph. Assumes that all items in a particular graph are in order in the tensor
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return torch.stack(src.split(sizes, dim), dim=0)
