import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot

from assessor_classes import AbstractAssessor
from card_pool import CardPool
from evaluation.environment import Environment
from metric_embedding.better_metric_embedding import ContrastiveCardIndexEmbeddingDataset, ContrastiveMetricEmbedding, \
    train
from metric_embedding.metric_embedding import load_data


def remove_from_dummy_data(df:pd.DataFrame, cube:pd.DataFrame, count) -> pd.DataFrame:
    """
    Remove all but the top "count" cards from the data, decided by the number of decks a card appeared in (only the cards with the most decks stay in the dataset)
    :param df: dataframe merging the decks and games dataframes on date and player
    :param cube: dataframe listing all the cards in the current cube. These cards cannot be removed from the dataset at any cost
    :param count: count of cards to stay in the dataset. If the number is equal to or smaller than the number of cards in the current cube, prunes the dataset to just those cards
    :return: pruned dataset
    """
    count = count - len(cube)

    count_per_card = df.sum(axis=0).rename('count').sort_values(ascending=False)
    # count_per_card = count_per_card[count_per_card.loc[:, 'card'] > threshold]
    flags = ~count_per_card.index.isin(cube.index)
    count_per_card = count_per_card[flags]
    selected = count_per_card.head(max(0, count))

    return df[selected.index.tolist() + cube.index.tolist()]


class MetricEmbeddingAssessor(AbstractAssessor):
    def __init__(self, cube, card_pool, removed, decks, games, model, num_closest_cards=0):
        super().__init__(cube, card_pool, removed)

        self.decks = decks
        self.games = games
        self.model = model
        if num_closest_cards <= 0 or num_closest_cards > len(decks.columns):
            self.num_closest_cards = len(decks.columns)
        else:
            self.num_closest_cards = num_closest_cards

    def _distance(self, a, b):
        return np.linalg.norm(a - b)

    def distance_between_cards(self, card1, card2):
        index1 = self.decks.columns.get_loc(card1)
        vector1 = one_hot(torch.Tensor([index1]).to(torch.int64), num_classes=len(self.decks.columns))[0].to(torch.float32)
        a = self.model(vector1).detach().numpy()

        index2 = self.decks.columns.get_loc(card2)
        vector2 = one_hot(torch.Tensor([index2]).to(torch.int64), num_classes=len(self.decks.columns))[0].to(torch.float32)
        b = self.model(vector2).detach().numpy()

        return self._distance(a, b)

    def calculate_card_score(self, cardname):
        if len(self.card_pool) == 0:
            return 0

        index = self.decks.columns.get_loc(cardname)
        vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.decks.columns))[0].to(torch.float32)
        a = self.model(vector).detach().numpy()

        closest_cards = []
        for k in self.card_pool:
            index = self.decks.columns.get_loc(k)
            pool_vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.decks.columns))[0].to(torch.float32)
            b = self.model(pool_vector).detach().numpy()
            closest_cards.append(self._distance(a, b))
            # distance += closest_cards[k]
            if len(closest_cards) > self.num_closest_cards:
                closest_cards.remove(min(closest_cards))

        distance = sum(closest_cards)

        avg = distance / len(self.card_pool)

        return 1/avg


# def load_data():
#     cube = pd.read_csv('../alahamaretov_arhiv/cube_copy.csv',
#                        usecols=['name', 'CMC', 'Type', 'Color'],
#                        dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
#     cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
#     cube.set_index('card', inplace=True)
#
#     decks = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
#                           sheet_name='Decks',
#                           usecols=['date', 'player', 'card'],
#                           dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})
#     decks.date = decks.date.apply(lambda x: x.date())
#     decks.set_index(['date', 'player'], inplace=True)
#
#     games = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
#                           sheet_name='Games',
#                           usecols=['date', 'player', 'opponent', 'wins', 'losses'],
#                           dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
#                                  'losses': 'int'})
#     games.date = games.date.apply(lambda x: x.date())
#     games.set_index(['date', 'player'], inplace=True)
#
#     return cube, decks, games


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    draft_num = 5
    epochs = 300
    epoch_to_save = epochs

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

    alpha = 0
    df2 = games.groupby(['date', 'player'])  # ['wins'].sum() #.apply(lambda x: x.wins/(x.wins+x.losses))
    deck_wins = df2['wins'].sum()
    deck_losses = df2['losses'].sum()
    deck_winrates = (deck_wins + alpha) / (deck_wins + deck_losses + 2 * alpha)

    y = deck_winrates.to_numpy()

    for k in range(360, len(df.columns), 10):
        df_pruned = remove_from_dummy_data(df, cube, k)

        X = df_pruned.to_numpy()

        dataset = ContrastiveCardIndexEmbeddingDataset(torch.Tensor(X), torch.Tensor(y))
        train_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=25,
        )

        to_save = {}
        for i in range(5, 146, 10):
            model = ContrastiveMetricEmbedding(k, i)
            # model.load_state_dict(torch.load("../metric_embedding/params.pt"))

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-4
            )

            for epoch in range(epochs):
                # print(f"Epoch: {epoch}")
                train_loss = train(model, optimizer, train_loader, device)
                if epoch==0 or (epoch+1) % epoch_to_save == 0:
                    # torch.save(model.state_dict(), "./params.pt")
                    print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")

            pool = CardPool(cube)
            removed = CardPool(cube)

            assessor = MetricEmbeddingAssessor(cube, pool, removed, df_pruned, deck_winrates, model)
            env = Environment(4, 3, 15, cube, pool, removed, assessor, draft_num=draft_num)

            try:
                while True:
                    env.simulate_pick()
            except StopIteration:
                print(f"Num_cards: {k}, embedding_dim: {i} Done")

            arr = []
            for index, card in enumerate(pool):
                # print(index+2, end=". ")
                if card==env.bot_pool[index]:
                    arr.append(1)
                else:
                    arr.append(0)
            to_save.update({i: arr})
        to_save = pd.DataFrame.from_dict(to_save)
        with pd.ExcelWriter(f"./results.xlsx", mode='a') as writer:
            to_save.to_excel(writer, sheet_name=f'{draft_num}_{k}')
            print(f"written {draft_num}_{k}")
        # print(to_save)