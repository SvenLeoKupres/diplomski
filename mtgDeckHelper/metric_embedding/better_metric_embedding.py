from random import choice

import numpy as np 
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torch.nn.utils import clip_grad_norm_

from metric_embedding import load_data

PRINT_LOSS_N = 10


class ContrastiveCardIndexEmbeddingDataset(Dataset):
    """
    Dataset which is used to generate batches and sample data combined with the card embedding network.
    """
    def __init__(self, data:torch.Tensor, targets:torch.Tensor, *args, **kwargs):
        """

        :param data: tensor. First dimension is the decks. Second dimension is the cards within the decks (1 if a card is present, 0 otherwise).
        :param targets: Its one and only dimension is decks. Average winrate of each deck. Potentially could upgrade to include the number of games with a deck in order to calculate average win-rates with weights
        """
        super().__init__(*args, **kwargs)

        self.data = data
        self.targets = targets

    def _sample(self, index:int) -> (torch.Tensor, float):
        """
        Used to randomly sample a datapoint.
        :param index: represents the index of the card in the second dimension of the dataset. That specific card is an anchor to sample the second card against
        :return: one-hot representation of a sampled card, as well as the average winrate of all the decks which contain both the sampled card and the card represented by the index
        """
        card = index
        while card==index:
            card = choice(range(self.data.shape[1]))

        deck_range_1 = self.data[:, index] == 1
        deck_range_2 = self.data[:, card] == 1

        deck_range = deck_range_2 * deck_range_1

        encoded = one_hot(torch.Tensor([card]).to(torch.int64), num_classes=self.data.shape[1])[0].to(torch.float32)
        if sum(deck_range) == 0:
            return encoded, 0

        winrate = self.targets[deck_range].mean().item()

        return encoded, winrate

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, index):
        """Indexes by card. Returns it and another random card"""
        x1 = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=self.data.shape[1])[0].to(torch.float32)

        # if self.data[:, index].sum() == 0:
        #     return x1, x1, 1

        x2, y = self._sample(index)

        return x1, x2, y


class ContrastiveMetricEmbedding(nn.Module):
    """
    A neural network which uses the contrastive loss function to train a metric embedding representation of cards
    """
    def __init__(self, card_num, emb_size=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.card_num = card_num

        self.layers = nn.Sequential(nn.Linear(self.card_num, self.card_num//2),
                                    nn.ReLU(),
                                    # nn.Linear(self.card_num//2, self.card_num//2),
                                    nn.Linear(self.card_num//2, emb_size))
        self.requires_grad_(True)

    def forward(self, data:torch.Tensor) -> torch.Tensor:
        """
        A pass through the neural network
        :param data: tensor with dimensions BATCH_SIZE x CARD_NUM
        :return: tensor with dimensions BATCH_SIZE x EMB_SIZE
        """
        return self.layers(data)

    def get_features(self, data:torch.Tensor) -> torch.Tensor:
        """
        Same as the forward method. Used for training
        :param data: tensor with dimensions BATCH_SIZE x CARD_NUM
        :return: tensor with dimensions BATCH_SIZE x EMB_SIZE
        """
        # Returns tensor with dimensions BATCH_SIZE x EMB_SIZE
        return self.forward(data)

    def loss(self, x_1:torch.Tensor, x_2:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """
        Calculates contrastive loss for a batch
        :param x_1: tensor with dimensions BATCH_SIZE x CARD_NUM
        :param x_2: tensor with dimensions BATCH_SIZE x CARD_NUM
        :param y: tensor with dimensions BATCH_SIZE, win-rates of card pairs (tensors x_1 and x_2)
        :return: average contrastive loss for a batch
        """
        x_1 = self.get_features(x_1)
        x_2 = self.get_features(x_2)

        d = (x_1 - x_2).pow(2).sum().sqrt()
        loss = d * y + (1-y) / d

        return loss.sum() / len(loss)


def train(model, optimizer, loader, device='cuda', to_print=False):
    losses = []
    model.train()
    for i, data in enumerate(loader):
        x_1, x_2, y = data
        optimizer.zero_grad()
        loss = model.loss(x_1.to(device), x_2.to(device), y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.cpu().item())
        if i % PRINT_LOSS_N == 0 and to_print:
            print(f"Iter: {i}, Mean Loss: {np.mean(losses):.3f}")
    return np.mean(losses)


if __name__ == '__main__':
    cube, decks, games = load_data()

    reshaped = pd.pivot_table(
        decks,
        index=decks.index,  # since the index already includes 'date' and 'player'
        columns='card',
        aggfunc=lambda x: 1,  # set to 1 where card was used
        fill_value=0
    )

    # Clean up column names
    reshaped.columns.name = None
    reshaped.columns = [f'{col}' for col in reshaped.columns]

    # First, make sure index is a MultiIndex, not a single column of tuples
    reshaped.index = pd.MultiIndex.from_tuples(reshaped.index, names=['date', 'player'])

    # Then reset it to get 'date' and 'player' as separate columns
    reshaped = reshaped.reset_index()

    cardnames = decks.loc[:, "card"].sort_values().unique()
    not_played = cube[~cube.index.isin(cardnames)].index

    reshaped[not_played] = 0

    df = reshaped.iloc[:, 2:]
    # df = remove_from_dummy_data(df, cube, 510)

    X = df.to_numpy()

    alpha = 0
    df2 = games.groupby(['date', 'player'])  # ['wins'].sum() #.apply(lambda x: x.wins/(x.wins+x.losses))
    deck_wins = df2['wins'].sum()
    deck_losses = df2['losses'].sum()
    deck_winrates = (deck_wins + alpha) / (deck_wins + deck_losses + 2 * alpha)

    y = deck_winrates.to_numpy()

    dataset = ContrastiveCardIndexEmbeddingDataset(torch.Tensor(X), torch.Tensor(y))
    # print(dataset[0])
    model = ContrastiveMetricEmbedding(len(dataset), 100)
    try:
        model.load_state_dict(torch.load("./params.pt"))
    except FileNotFoundError:
        print("No saved file found")
    except RuntimeError:
        pass

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

    epochs = 10000
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, optimizer, train_loader, device, to_print=True)
        if epoch % epoch_to_save == 0:
            torch.save(model.state_dict(), "./params.pt")
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")