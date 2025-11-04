import pandas as pd

from metric_embedding import load_data

import numpy as np


class InputSpace:
    def __init__(self, data):
        self.data = data

    def transform(self):
        raise NotImplementedError

    def input_size(self):
        raise NotImplementedError

class OneHotInputSpace(InputSpace):
    def __init__(self, data):
        super().__init__(data)

    def transform(self):
        pass

    def input_size(self):
        return len(self.data)

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