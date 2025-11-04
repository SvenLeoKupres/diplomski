import math
from random import choice, randint

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_


PRINT_LOSS_N = 10


def load_data():
    cube = pd.read_csv('../alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    decks = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Decks',
                          usecols=['date', 'player', 'card'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})
    decks.date = decks.date.apply(lambda x: x.date())
    decks.set_index(['date', 'player'], inplace=True)

    games = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Games',
                          usecols=['date', 'player', 'opponent', 'wins', 'losses'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
                                 'losses': 'int'})
    games.date = games.date.apply(lambda x: x.date())
    games.set_index(['date', 'player'], inplace=True)

    return cube, decks, games


def get_random_tensor_with_one_at(index, tensor_list):
    # Filter tensors where any row has a 1 at the specified column index
    filtered = [t for t in tensor_list if t[:, index].any()]

    if not filtered:
        return None
        # raise ValueError(f"No tensor contains a 1 at index {index}.")

    return choice(filtered)


def elongate_vector(vector):
    # matrix = [[1 if i==k and vector[i]==1 else 0 for i in range(len(vector))] for k in range(len(vector))]
    # # return matrix
    #
    # k = 0
    # while k < len(matrix):
    #     if all(not x for x in matrix[k]):
    #         matrix.remove(matrix[k])
    #         k -= 1
    #     k += 1
    #
    # return matrix

    indices = torch.nonzero(vector, as_tuple=True)[0]

    # Create one-hot encodings
    return one_hot(indices, num_classes=vector.size(0))


class CardIndexEmbeddingDataset(Dataset):
    def __init__(self, data, targets, calculate_winrate='mean', threshold=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.targets = targets

        # method to calculate card winrate, for now either 'mean' or 'weighted'
        self.calculate_winrate = calculate_winrate

        # Threshold for when a card is considered positive
        self.threshold = threshold

    def _sample(self, index, check_function):
        possible = self.data[self.data[:, index] == 1]
        # if len(possible) == 1:
        #     winrate = self.targets[self.data[:, index] == 1].mean().item()
        #     if check_function(winrate, self.threshold):
        #         return \
        #         one_hot(torch.Tensor([index]).to(torch.int64), num_classes=self.data.shape[1])[
        #             0].to(torch.float32), 0
        #     else:
        #         card = choice(torch.nonzero(possible[0], as_tuple=True)[0])
        #         return \
        #         one_hot(torch.Tensor([card]).to(torch.int64), num_classes=self.data.shape[1])[0].to(
        #             torch.float32), winrate

        deck_index = choice(range(len(possible)))
        tried_decks = []
        while len(tried_decks) < len(possible):
            while deck_index in tried_decks:
                deck_index = choice(range(len(possible)))
            deck = possible[deck_index]
            tried_cards = [index]
            while len(tried_cards) < len(torch.nonzero(deck, as_tuple=True)[0]):
                card = choice(torch.nonzero(deck, as_tuple=True)[0])
                while card.item() in tried_cards:
                    card = choice(torch.nonzero(deck, as_tuple=True)[0])

                deck_range_1 = self.data[:, index] == 1
                deck_range_2 = self.data[:, card] == 1

                deck_range = deck_range_2 * deck_range_1
                winrate = None
                if self.calculate_winrate == 'mean':
                    winrate = self.targets[deck_range].mean().item()
                elif self.calculate_winrate == 'weighted':
                    winrate = (self.targets[deck_range,0]*self.targets[deck_range,1]).sum()/self.targets[deck_range,1].sum()
                if check_function(winrate, self.threshold):
                    card_pick = \
                    one_hot(torch.Tensor([card]).to(torch.int64), num_classes=self.data.shape[1])[0].to(
                        torch.float32)
                    return card_pick, winrate
                tried_cards.append(card.item())
            tried_decks.append(deck_index)

        return one_hot(torch.Tensor([index]).to(torch.int64), num_classes=self.data.shape[1])[0].to(
                        torch.float32), 0

    def _sample_negative(self, index):
        check_function = lambda a, b: a<b
        return self._sample(index, check_function)

    def _sample_positive(self, index):
        check_function = lambda a, b: a>=b
        return self._sample(index, check_function)

    def __getitem__(self, index):
        """Indexes by card. If index is of a card which was never played, return null.
        Otherwise, returns it, a positive and a negative sample with average winrate of both the anchor and sample cards in decks"""
        # if self.data[:, index].sum() == 0:
        #     return None

        anchor = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=self.data.shape[1])[0].to(torch.float32)

        if self.data[:, index].sum() == 0:
            return anchor, anchor, anchor, 0, 0

        positive, target_positive = self._sample_positive(index)
        negative, target_negative = self._sample_negative(index)

        return anchor, positive, negative, target_positive, target_negative

    def __len__(self):
        return self.data.shape[1]


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, card_num, emb_size=32, margin=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.card_num = card_num
        self.margin = margin

        self.layers = nn.Sequential(nn.Linear(self.card_num, self.card_num//2),
                                    nn.ReLU(),
                                    # nn.Linear(self.card_num//2, self.card_num//2),
                                    # nn.ReLU(),
                                    nn.Linear(self.card_num//2, emb_size))
        self.requires_grad_(True)

    def forward(self, data):
        return self.layers(data)

    def get_features(self, data):
        """Returns tensor with dimensions BATCH_SIZE, EMB_SIZE"""
        x = self.forward(data)
        return x

    def loss(self, anchor, positive, negative, t_p, t_n):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # return nn.TripletMarginLoss().forward(anchor=a_x, positive=p_x, negative=n_x)
        loss = (a_x - p_x).pow(2).sum().sqrt() * t_p - (a_x - n_x).pow(2).sum().sqrt() * (1-t_n) + self.margin
        loss[loss<0] = 0
        return loss.sum() / len(loss)


def train(model, optimizer, loader, device='cuda'):
    losses = []
    model.train()
    for i, data in enumerate(loader):
        anchor, positive, negative, t_p, t_n = data
        optimizer.zero_grad()
        loss = model.loss(anchor.to(device), positive.to(device), negative.to(device), t_p, t_n)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.cpu().item())
        if i % PRINT_LOSS_N == 0:
            print(f"Iter: {i}, Mean Loss: {np.mean(losses):.3f}")
    return np.mean(losses)


def create_dataset(calculate_winrate='mean'):
    cube, decks, games = load_data()
    # df = decks.merge(games, on=["date", "player"])

    reshaped = pd.pivot_table(
        decks,
        index=decks.index,  # since the index already includes 'date' and 'player'
        columns='card',
        aggfunc=lambda x: 1,  # set to 1 where card was used
        fill_value=0
    )

    # Clean up column names
    # reshaped.columns.name = None
    # reshaped.columns = [f'{col}' for col in reshaped.columns]

    # First, make sure index is a MultiIndex, not a single column of tuples
    # reshaped.index = pd.MultiIndex.from_tuples(reshaped.index, names=['date', 'player'])

    # Then reset it to get 'date' and 'player' as separate columns
    # reshaped = reshaped.reset_index()

    cardnames = decks.loc[:, "card"].sort_values().unique()
    not_played = cube[~cube.index.isin(cardnames)].index

    reshaped[not_played] = 0

    df = reshaped.iloc[:, 2:]

    X = df.to_numpy()

    alpha = 0
    df2 = games.groupby(['date', 'player'])  # ['wins'].sum() #.apply(lambda x: x.wins/(x.wins+x.losses))
    deck_wins = df2['wins'].sum()
    deck_losses = df2['losses'].sum()
    deck_winrates = (deck_wins + alpha) / (deck_wins + deck_losses + 2 * alpha)

    y = None
    if calculate_winrate == 'mean':
        y = deck_winrates.to_numpy()
    elif calculate_winrate == 'weighted':
        deck_total_games = deck_wins + deck_losses
        result = pd.concat([deck_winrates, deck_total_games], axis=1, ignore_index=True)
        y = result.to_numpy()

    return CardIndexEmbeddingDataset(torch.Tensor(X), torch.Tensor(y), calculate_winrate)

    # better to do it lazy
    # new_X = []
    # for k in X:
    #     new_X.append(elongate_vector(torch.Tensor(k)))
    #
    # # new_X = torch.Tensor(new_X) # torch.tensor(new_X)
    # pass

    # fin_index = int(0.8*len(X))
    #
    # train_dataset = CardIndexEmbeddingDataset(torch.Tensor(X)[:fin_index], torch.Tensor(y)[:fin_index])
    # test_dataset = CardIndexEmbeddingDataset(torch.Tensor(X)[fin_index:], torch.Tensor(y)[fin_index:])


if __name__ == '__main__':
    # vector = torch.tensor([1, 0, 0, 0, 1, 0, 1])
    # print(elongate_vector(vector))

    dataset = create_dataset(calculate_winrate='weighted')
    model = SimpleMetricEmbedding(len(dataset), 100)
    # try:
    #     model.load_state_dict(torch.load("./params.pt"))
    # except FileNotFoundError:
    #     print("No saved file found")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=25,
        #drop_last=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epoch_to_save = 20

    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, optimizer, train_loader, device)
        if epoch % epoch_to_save == 0:
            torch.save(model.state_dict(), "./params.pt")
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")

