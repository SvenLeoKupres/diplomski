import pandas as pd

from metric_embedding import load_data

import numpy as np

import torch
from torch.nn.functional import one_hot


class InputSpace:
    """
    A way to transform the inputs (so they do not have to be just one-hot encoded), in order to shrink it
    """
    def __init__(self, size:int):
        """
        :param size: size of the vector needed
        """
        self.size = size

    def transform(self, index:int) -> torch.Tensor:
        """
        performs the transformation
        :param index: index of the card
        :return: a tensor of size (size)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        :return: size (dimension) of the input space
        """
        return self.size

class OneHotInputSpace(InputSpace):
    """
    Transforms a card into a one-hot encoded tensor
    """
    def __init__(self, size):
        super().__init__(size)

    def transform(self, index):
        return one_hot(torch.Tensor([index]).to(torch.int64), num_classes=self.size)[0].to(torch.float32)

class BinaryInputSpace(InputSpace):
    """
    Transforms data to binary encoded data.
    For instance, 10000 (fifth bit is 1, all others 0) would be transformed into 101 (5 in binary), with padding to satisfy the "size" input
    """
    def __init__(self, size):
        super().__init__(np.log2(size))
        if self.size%1>0:
            self.size = int(self.size+1)
        else:
            self.size = int(self.size)

    def transform(self, index):
        binary = bin(k)[2:].zfill(length)
        return torch.Tensor(binary).to(torch.float32)

if __name__=='__main__':
    cube, decks, games = load_data()

    cardnames = decks.loc[:, "card"].sort_values().unique().tolist()
    not_played = cube[~cube.index.isin(cardnames)].index.tolist()

    cardnames += not_played

    cardnames = pd.DataFrame(cardnames)
    # cardnames['binary']=bin(cardnames.index.tolist())

    length = np.log2(cardnames.size)
    if length%1>0:
        length = int(length+1)
    else:
        length = int(length)
    binary = [bin(k)[2:].zfill(length) for k in range(cardnames.size)]

    binary_dict = {}
    for k in range(length):
        binary_dict[k] = [i[k] for i in binary]
    binary = pd.DataFrame(binary_dict)

    pass