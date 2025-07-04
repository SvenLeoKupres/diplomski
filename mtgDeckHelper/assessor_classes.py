import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch.nn.functional import one_hot

import paretto_fronts
from card_pool import CardPool
# from display_classes import Window
from metric_embedding.metric_embedding import SimpleMetricEmbedding


def loadDataFromFiles():
    decks = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Decks',
                          usecols=['date', 'player', 'card'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})


    decks.date = decks.date.apply(lambda x: x.date())
    decks.set_index(['date', 'player'], inplace=True)

    games = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Games',
                          usecols=['date', 'player', 'opponent', 'wins', 'losses'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
                                 'losses': 'int'})
    games.date = games.date.apply(lambda x: x.date())
    games.set_index(['date', 'player'], inplace=True)

    return decks, games


def smooth_winrate(alpha=1):
    def winrate(g: pd.DataFrame):
        return (g["wins"].sum() + alpha) / (g["wins"].sum() + g["losses"].sum() + 2*alpha)

    return winrate

def extract_game_count(decks, games):
    df = decks.merge(games, on=["date", "player"])
    return df.groupby(["card"]).apply(
        lambda g: g["wins"].sum() + g["losses"].sum()
    ).rename("game_count")


def extract_deck_count(decks):
    return decks.groupby(["card"])['card'].count().rename("deck_count")


def add_color_bonus(assessor, cardname, color_num=1):
    colors = assessor.card_pool.get_color_dict()
    card_color = assessor.cube.loc[cardname, "color"]
    if type(card_color) is float:
        return 0

    color_bonus = 0
    for color in card_color:
        color_bonus += colors[color]

    return color_bonus*color_num

def static_calculate_card_score(assessor, cardname):
    """Calculates card score based on the assessor data about the cardname's Paretto front and how many colors in the picked pool it shares a color with"""
    score = len(assessor.fronts)

    k = 0
    while k < score and cardname not in assessor.fronts[k]:
        k += 1

    return score - k


def static_form_basic_text(score, color_bonus):
    text = f"Basic score: {score}\n"
    text += f"Color bonus: {color_bonus}\n\n"
    text += f"Total: {score + color_bonus}"

    return text


class AbstractAssessor:
    """based on provided context, calculates the score for a given card"""
    def __init__(self, cube, card_pool, removed):
        self.cube = cube
        self.card_pool = card_pool
        self.removed = removed

    def result_string(self, cardname):
        return static_form_basic_text(0, add_color_bonus(self, cardname, 1))

    def calculate_card_score(self, cardname):
        return add_color_bonus(self, cardname, 1)


class BasicAssessor(AbstractAssessor):
    """Calculates the score for a given card based on base winrate.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num=1, alpha=1, relevant_digits = 2):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()

        df = decks.merge(games, on=["date", "player"])

        card_wr = df.groupby(["card"]).apply(smooth_winrate(alpha)).rename("winrate")

        cardnames = decks.loc[:, "card"].sort_values().unique()
        not_played = cube[~cube.index.isin(cardnames)].loc[:, []]
        card_wr = pd.concat([card_wr, not_played], axis=1)
        card_wr.fillna(value=0.5, inplace=True)

        self.card_wr = (card_wr * 10 ** relevant_digits).astype('int32').iloc[:, 0]

    def result_string(self, cardname):
        return static_form_basic_text(self.card_wr.at[cardname], add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return self.card_wr.at[cardname] + add_color_bonus(self, cardname, self.color_num)


class ParettoFrontAssessor(AbstractAssessor):
    """Calculates the score for a given card based on base winrate.
        Uses Paretto fronts (the criteria are the amount of decks a card has been played in and card winrate)
        to divide the cards into tiers.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num=1, alpha=1):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()

        df = decks.merge(games, on=["date", "player"])

        card_wr = df.groupby(["card"]).apply(smooth_winrate(alpha)).rename("winrate")

        cardnames = decks.loc[:, "card"].sort_values().unique()
        not_played = cube[~cube.index.isin(cardnames)].loc[:, []]
        card_wr = pd.concat([card_wr, not_played], axis=1)
        card_wr.fillna(value=0.5, inplace=True)

        count = extract_deck_count(decks)
        count = pd.concat([count, not_played], axis=1)
        count.fillna(value=0, inplace=True)

        card_stats = pd.concat([count, card_wr], axis=1)  # .rename(columns={"0": "winrate"})

        self.fronts = paretto_fronts.divide_into_fronts(card_stats)

    def result_string(self, cardname):
        return static_form_basic_text(static_calculate_card_score(self, cardname), add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return static_calculate_card_score(self, cardname) + add_color_bonus(self, cardname, self.color_num)


def fixed_effects_model(df, alpha):
    player_wr = df.groupby("player").apply(smooth_winrate(alpha)).rename("winrate")
    daily_player_wr = df.groupby(["date", "player"]).apply(smooth_winrate(alpha)).rename("daily_winrate").to_frame()
    daily_player_wr['winrate'] = daily_player_wr.index.get_level_values('player').map(player_wr)
    daily_player_wr["lift"] = daily_player_wr["daily_winrate"] - daily_player_wr["winrate"]

    card_wr = df.groupby(["card"]).apply(smooth_winrate(alpha)).rename("winrate")
    daily_card_wr = df.groupby(["date", "card"]).apply(smooth_winrate(alpha)).rename("daily_winrate").to_frame()
    daily_card_wr["winrate"] = daily_card_wr.index.get_level_values('card').map(card_wr)
    daily_card_wr["lift"] = daily_card_wr["daily_winrate"] - daily_card_wr["winrate"]

    y = np.nan_to_num(daily_player_wr.loc[:, "lift"].unstack().to_numpy(), nan=0)
    X = np.nan_to_num(daily_card_wr.loc[:, "lift"].unstack().to_numpy(), nan=0)
    return LinearRegression().fit(X, y)

class FixedEffectsAssessor(AbstractAssessor):
    """Uses a Fixed Effects Model (in order to normalize the card winrate)
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num=1, alpha=1, relevant_digits=3):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()

        df = decks.merge(games, on=["date", "player"])

        model = fixed_effects_model(df, alpha)

        # count = extract_deck_count(decks)

        diff = model.coef_.sum(axis=0)
        cardnames = decks.loc[:, "card"].sort_values().unique()
        card_lift = pd.Series(data=diff, index=cardnames).rename("winrate")

        # card_stats = pd.concat([count, card_lift], axis=1)

        not_played = cube[~cube.index.isin(cardnames)]
        self.card_lift = pd.concat([card_lift, not_played], axis=1).loc[:, 'winrate']
        self.card_lift.fillna(value=0, inplace=True)
        self.card_lift = (self.card_lift*10**relevant_digits).astype("int32")

    def result_string(self, cardname):
        return static_form_basic_text(self.card_lift[cardname], add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return self.card_lift.at[cardname] + add_color_bonus(self, cardname, self.color_num)


class FixedEffectsParettoFrontAssessor(AbstractAssessor):
    """Uses a Fixed Effects Model (in order to normalize the card winrate)
        and Paretto Fronts (the criteria are the amount of decks a card has been played in and card winrate)
        to divide the cards into tiers.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num=1, alpha=1):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()

        df = decks.merge(games, on=["date", "player"])

        model = fixed_effects_model(df, alpha)

        count = extract_deck_count(decks)

        diff = model.coef_.sum(axis=0)
        cardnames = decks.loc[:, "card"].sort_values().unique()
        card_lift = pd.Series(data=diff, index=cardnames).rename("winrate")

        card_stats = pd.concat([count, card_lift], axis=1)

        not_played = cube[~cube.index.isin(cardnames)]
        card_stats = pd.concat([card_stats, not_played], axis=1).iloc[:, :2]
        card_stats.fillna(value=0, inplace=True)

        self.fronts = paretto_fronts.divide_into_fronts(card_stats)

    def result_string(self, cardname):
        return static_form_basic_text(static_calculate_card_score(self, cardname), add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return static_calculate_card_score(self, cardname) + add_color_bonus(self, cardname, self.color_num)


class SimpleMetricEmbeddingAssessor(AbstractAssessor):
    def __init__(self, cube, card_pool, removed, num_closest_cards=3):
        #TODO remove hardcoded embedding_dim and input_dim
        super().__init__(cube, card_pool, removed)

        decks, games = loadDataFromFiles()

        cardnames = decks.loc[:, "card"].sort_values().unique().tolist()
        not_played = cube[~cube.index.isin(cardnames)].index.tolist()

        self.card_list = cardnames + not_played

        self.model = SimpleMetricEmbedding(len(cardnames) + len(not_played), 100)
        try:
            self.model.load_state_dict(torch.load("metric_embedding/params.pt"))
        except FileNotFoundError:
            print("No saved file found")
            raise FileNotFoundError

        self.num_closest_cards = num_closest_cards


    def _distance(self, a, b):
        return np.linalg.norm(a - b)


    def distance_between_cards(self, card1, card2):
        index1 = self.card_list.index(card1)
        vector1 = one_hot(torch.Tensor([index1]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
        a = self.model(vector1).detach().numpy()

        index2 = self.card_list.index(card2)
        vector2 = one_hot(torch.Tensor([index2]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
        b = self.model(vector2).detach().numpy()

        return self._distance(a, b)


    def result_string(self, cardname):
        text = ""

        if len(self.card_pool) == 0:
            return text + f"Average distance from deck: {0:.4f}"

        index = self.card_list.index(cardname)
        vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
        a = self.model(vector).detach().numpy()

        closest_cards = dict()
        distance = 0
        for k in self.card_pool:
            index = self.card_list.index(k)
            pool_vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
            b = self.model(pool_vector).detach().numpy()
            closest_cards[k] = self._distance(a, b)
            distance += closest_cards[k]
            # if len(closest_cards[k]) < num_closest_cards:
            # closest_cards[k] = distance
            if len(closest_cards) > self.num_closest_cards:
                name = k
                for i in closest_cards.keys():
                    if closest_cards[i] > closest_cards[name]:
                        # maximum = closest_cards[i]
                        name = i
                del closest_cards[name]

        avg = distance / len(self.card_pool)

        text += f"Average distance from deck: {avg:.4f}\n\n"

        text += "Best card synergies:\n"
        # card_distances = closest_cards
        for k in closest_cards.keys():
            text += f"{k}, distance: {closest_cards[k]: .4f}\n"

        return text

    def calculate_card_score(self, cardname):
        if len(self.card_pool) == 0:
            return 0

        index = self.card_list.index(cardname)
        vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
        a = self.model(vector).detach().numpy()

        closest_cards = dict()
        distance = 0
        for k in self.card_pool:
            index = self.card_list.index(k)
            pool_vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
            b = self.model(pool_vector).detach().numpy()
            closest_cards[k] = self._distance(a, b)
            distance += closest_cards[k]
            if len(closest_cards) > self.num_closest_cards:
                name = k
                for i in closest_cards.keys():
                    if closest_cards[i] > closest_cards[name]:
                        name = i
                del closest_cards[name]

        avg = distance / len(self.card_pool)

        return 1/avg


class CompositeAssessor(AbstractAssessor):
    def __init__(self, cube, card_pool, removed, assessors: {}):
        super().__init__(cube, card_pool, removed)

        self.assessors = assessors

    def result_string(self, cardname):
        text = ""
        for k in self.assessors.keys():
            text += f"{k}\n{self.assessors[k].result_string(cardname)}\n\n\n"
        return text

    def calculate_card_score(self, cardname):
        raise NotImplementedError


# removed due to circular imports (Window from display_classes)
# if __name__ == '__main__':
#     cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
#                        usecols=['name', 'CMC', 'Type', 'Color'],
#                        dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
#     cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
#     cube.set_index('card', inplace=True)
#
#     card_pool = CardPool(cube)
#     removed_cards = CardPool(cube)
#
#     assessor = FixedEffectsAssessor(cube, card_pool, removed_cards)
#
#     card_list = cube.index.tolist()
#
#     root = Window(card_list, 2, card_pool, removed_cards, assessor)
#     height = 1000
#     width = 1300
#     geometry = str(width) + 'x' + str(height)
#     root.geometry(geometry)
#
#     card_data = root.get_pack_data()
#     # root.add_button_listener(lambda: print(card_data))
#
#     root.mainloop()