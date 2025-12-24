import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot

import paretto_fronts
from card_pool import CardPool
from demos.fixed_effects_model import fe_model
from metric_embedding.metric_embedding import SimpleMetricEmbedding


def loadDataFromFiles() -> [pd.DataFrame, pd.DataFrame]:
    """

    :return: decks and games from the excel file alahamaretov_arhiv/shoebox.xlsx
    """
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


def smooth_winrate(alpha:float=1):
    """
    A function working as a functional interface for winrate calculation and smoothing
    :param alpha: the smoothing coefficient, default 1
    :return: a function which accepts a dataframe with columns "wins" and "losses" and returns a new column representing the winrate,
    smoothed with the alpha coefficient
    """

    def winrate(g: pd.DataFrame):
        return (g["wins"].sum() + alpha) / (g["wins"].sum() + g["losses"].sum() + 2*alpha)

    return winrate

def extract_game_count(decks, games) -> pd.DataFrame:
    """

    :param decks: Pandas dataframe containing the decks played
    :param games: Pandas dataframe listing match results
    :return: Total number of games played for each card
    """
    df = decks.merge(games, on=["date", "player"])
    return df.groupby(["card"]).apply(
        lambda g: g["wins"].sum() + g["losses"].sum()
    ).rename("game_count")


def extract_deck_count(decks) -> pd.Series:
    """
    Calculates the number of decks each card is present in
    :param decks: dataframe with at least a "card" column
    :return: a series with "card" as the key and a column "deck_count"
    """
    return decks.groupby(["card"])['card'].count().rename("deck_count")


def add_color_bonus(assessor, cardname:str, color_num:float=1) -> float:
    """
    Calculates the additional score for a card based on its colors and the colors of drafted cards
    :param assessor: assessor used to calculate score. It contains both the data on the drafted cards, and the cards in the cube and their colors
    :param cardname: name of the considered card
    :param color_num: strength of the color bonus, default 1, that is 1 additional point for each card of the considered card's colors in the pool
    (multicolored cards in the pool are only counted once)
    :return: the considered card's bonus score
    """
    colors = assessor.card_pool.get_color_dict()
    card_color = assessor.cube.loc[cardname, "color"]
    if type(card_color) is float:   # if card is colorless
        return 0

    color_bonus = 0
    for color in card_color:
        color_bonus += colors[color]

    return color_bonus*color_num

def static_calculate_pareto_card_score(assessor, cardname:str) -> int:
    """
    Calculates card score based on the assessor data about the cardname's Paretto front and how many colors in the picked pool it shares a color with
    :param assessor: assessor used to calculate score
    :param cardname: name of the considered card
    :return: the card score based on the pareto front"""

    return max(assessor.fronts['front']) - assessor.fronts.at[cardname, 'front']


def static_form_basic_text(score:float, color_bonus:float) -> str:
    """

    :param score: Basic card score
    :param color_bonus: Color bonus for the card
    :return: Basic stringified card score
    """
    text = f"Basic score: {score}\n"
    text += f"Color bonus: {color_bonus}\n\n"
    text += f"Total: {score + color_bonus}"

    return text


class AbstractAssessor:
    """based on provided context, calculates the score for a given card"""
    def __init__(self, cube:pd.DataFrame, card_pool:CardPool, removed:CardPool):
        self.cube = cube
        self.card_pool = card_pool
        self.removed = removed

    def result_string(self, cardname: str) -> str:
        """
        Stringifies the card score
        :param cardname: name of the considered card
        :return: String representation of the card score
        """
        return static_form_basic_text(0, add_color_bonus(self, cardname, 1))

    def calculate_card_score(self, cardname) -> float:
        """
        Calculates the card score for a given card
        :param cardname: name of the considered card
        :return: score of the considered card
        """
        return add_color_bonus(self, cardname, 1)


class BasicAssessor(AbstractAssessor):
    """Calculates the score for a given card based on base (or smoothed) winrate.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num:float=1, alpha:float=1, relevant_digits=2):
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
    def __init__(self, cube, card_pool, removed, color_num:float=1, alpha:float=1):
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
        return static_form_basic_text(static_calculate_pareto_card_score(self, cardname), add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return static_calculate_pareto_card_score(self, cardname) + add_color_bonus(self, cardname, self.color_num)


class FixedEffectsAssessor(AbstractAssessor):
    """Uses a Fixed Effects Model in order to normalize the card winrate accounting for player skill.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num:float=1, alpha:float=1, relevant_digits=3):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()

        df = decks.merge(games, on=["date", "player"])

        model = fe_model(df, alpha)

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
    """Uses a Fixed Effects Model in order to normalize the card winrate accounting for player skill.
        Also uses Paretto Fronts (the criteria are the amount of decks a card has been played in and normalized card winrate)
        to divide the cards into tiers.
        On top of that, gives an advantage to colors which are already in the card pool"""
    def __init__(self, cube, card_pool, removed, color_num:float=1, alpha:float=1):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()
        df = decks.merge(games, on=["date", "player"])

        model = fe_model(df, alpha)

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
        return static_form_basic_text(static_calculate_pareto_card_score(self, cardname), add_color_bonus(self, cardname, self.color_num))

    def calculate_card_score(self, cardname):
        return static_calculate_pareto_card_score(self, cardname) + add_color_bonus(self, cardname, self.color_num)


class AverageDeckCMCAssessor(AbstractAssessor):
    #TODO train an assessor that averages deck cmc, and then trains a linear regression model with deck winrates as the target.
    # Assessor will work so that it wants the created deck to be as close to the best average deck CMC as possible
    def __init__(self, cube, card_pool, removed, color_num:float=1):
        super().__init__(cube, card_pool, removed)

        self.color_num = color_num

        decks, games = loadDataFromFiles()
        df = decks.merge(games, on=["date", "player"])

    def result_string(self, cardname):
        pass

    def calculate_card_score(self, cardname):
        pass




class SimpleMetricEmbeddingAssessor(AbstractAssessor):
    """
    Uses trained metric embeddings of cards in order to evaluate the distance between the considered and drafted cards
    """
    def __init__(self, cube, card_pool, removed, num_closest_cards=3):
        # maybe remove hardcoded embedding_dim and input_dim - problem is having to train it when starting...
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

        distance = 0
        for k in self.card_pool:
            index = self.card_list.index(k)
            pool_vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
            b = self.model(pool_vector).detach().numpy()
            distance += self._distance(a, b)

        avg = distance / len(self.card_pool)

        # if -10e-6<distance<10e-6:
        #     return 1000

        return 1/avg


class MetricEmbeddingAssessorClosest(SimpleMetricEmbeddingAssessor):
    """
    Works the same as the SimpleMetricEmbeddingAssessor except for it uses the distance to the closest card in the card pool instead of the average distance
    """
    def __init__(self, cube, card_pool, removed, num_closest_cards=3):
        super().__init__(cube, card_pool, removed, num_closest_cards)

    def result_string(self, cardname):
        return super().result_string(cardname)

    def calculate_card_score(self, cardname):
        if len(self.card_pool) == 0:
            return 0

        index = self.card_list.index(cardname)
        vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
        a = self.model(vector).detach().numpy()

        best_distance = np.inf
        for k in self.card_pool:
            index = self.card_list.index(k)
            pool_vector = one_hot(torch.Tensor([index]).to(torch.int64), num_classes=len(self.card_list))[0].to(torch.float32)
            b = self.model(pool_vector).detach().numpy()
            distance = self._distance(a, b)
            if best_distance > distance:
                best_distance = distance

        # if -10e-6<distance<10e-6:
        #     return 1000

        return 1/best_distance


class CompositeAssessor(AbstractAssessor):
    """
    Uses several asessors at once. Cannot calculate card score since it cannot specify how to prioritize assessors. Provides ability to easily show the output of several assessors
    """
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


# removed due to circular imports
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