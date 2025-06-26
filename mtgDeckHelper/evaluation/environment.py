import math
import random
import pandas as pd

from assessor_classes import BasicAssessor, ParettoFrontAssessor, FixedEffectsAssessor, \
    FixedEffectsParettoFrontAssessor, SimpleMetricEmbeddingAssessor
from card_pool import CardPool


class TextBuilder:
    def __init__(self, num_players, num_rounds, card_pool, num_cards_per_pack=15):
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.card_pool = card_pool
        self.removed = removed
        self.num_cards_per_pack = num_cards_per_pack
        self.card_pool.add_change_listener(self.update_card_pool)
        self.text = ""

    def update_card_pool(self):
        self.text += f"{self.card_pool.cards[-1]}\n"

    def save(self, path='./evaluation/draft'):
        with open(path, 'w') as f:
            f.write(self.text)


def save_pack(packs, num_players, num_rounds, num_cards_per_pack, path='./packs'):
    data = f"{num_players};{num_rounds};{num_cards_per_pack};{packs[0]}"
    for index in range(1, len(packs)):
        data += f";{packs[index]}"
    with open(path, 'w') as f:
        f.write(data)

def load_pack(path = './packs'):
    try:
        with open(path, 'r') as f:
            packs = f.read()
            data = packs.split(";")
            return int(data[0]), int(data[1]), int(data[2]), data[3:]
    except FileNotFoundError:
        raise FileNotFoundError("File doesn't exist")

def load_draft(path='./draft'):
    with open(path, 'r') as f:
        text = f.read()
        text = text.split("\n")[3:]
        text = [k.split("removed:") for k in text]
        text = [[i.split("added:") for i in k] for k in text]
        to_remove = []
        to_add = []
        for k in text:
            if len(k) == 1:
                to_add.append(k[0][1].strip())
            else:
                k = k[1]
                removal = k[0].split("; ")
                for i in range(0, len(removal)-1):
                    to_remove.append(removal[i].strip())
                if len(k) == 2:
                    to_add.append(k[1].strip())

    return to_add, to_remove


class Environment:
    def __init__(self, num_players, num_rounds, num_cards_per_pack, cube, card_pool, removed, assessor, mode="load file", save=False, draft_num=1):
        self.cube = cube
        self.assessor = assessor
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.num_cards_per_pack = num_cards_per_pack
        self.save = save

        self.pool = card_pool
        self.removed = removed
        if mode == "load file":
            self.num_players, self.num_rounds, self.num_cards_per_pack, shuffled_packs = load_pack(f"../old_drafts/packs{draft_num}")
        else:
            shuffled_packs = cube.index.tolist()
            shuffled_packs = random.sample(shuffled_packs, num_cards_per_pack * num_players * num_rounds)

            if save:
                save_pack(shuffled_packs, self.num_players, self.num_rounds, self.num_cards_per_pack)

        self.packs = [[[shuffled_packs[k * num_cards_per_pack * num_players + i * num_cards_per_pack + j] for j in range(num_cards_per_pack - i)] for i in range(num_players)] for k in range(num_rounds)]

        self.current_round = 0
        self.current_pack = 0
        self.current_pick = 0

        player, other_players = load_draft(f"../old_drafts/draft{draft_num}")
        self.player = []
        self.other_players = []
        for k in range(self.num_rounds):
            shift_1 = k*(self.num_cards_per_pack-self.num_players)*(self.num_players-1)
            self.player.append([])
            for i in range(self.num_cards_per_pack):
                self.player[k].append(player[k*self.num_cards_per_pack + i])

            self.other_players.append([])
            for i in range(self.num_players):
                self.other_players[k].append([])
            for i in range(self.num_cards_per_pack-self.num_players):
                shift_2 = i*(self.num_players-1)
                self.other_players[k].append([])
                for j in range(self.num_players-1):
                    self.other_players[k][i+self.num_players].append(other_players[shift_1 + shift_2 + j])

        self.done = False

        self.bot_pool = CardPool(cube)
        if save:
            self.text_builder = TextBuilder(self.num_players, self.num_rounds, self.bot_pool, self.num_cards_per_pack)

    def simulate_pick(self):
        if self.done:
            raise StopIteration

        current_pack = self.packs[self.current_round][self.current_pack]

        for k in self.other_players[self.current_round][self.current_pick]:
            current_pack.remove(k)
            self.removed.append(k)

        max_score = -math.inf
        best_index = 0
        for index, k in enumerate(current_pack):
            score = self.assessor.calculate_card_score(k)
            if score > max_score:
                max_score = score
                best_index = index
        self.bot_pool.append(current_pack[best_index])

        current_pack.remove(self.player[self.current_round][self.current_pick])
        self.pool.append(self.player[self.current_round][self.current_pick])

        if len(current_pack) == 0:
            if self.current_round == self.num_rounds - 1:
                self.done = True
                if self.save:
                    self.text_builder.save()
                raise StopIteration
            self.current_round += 1
            self.current_pack = 0
            self.current_pick = 0
        else:
            self.current_pick += 1
            self.current_pack = (self.current_pack + 1) % self.num_players


if __name__ == '__main__':
    cube = pd.read_csv('./alahamaretov_arhiv/cube_copy.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    by_hand = False

    df = {}
    for alpha in range(0, 6): # ne zaboravi +1
        df_1 = {}
        print(f"alpha={alpha}")
        for color_num in range(0, 81): # ne zaboravi +1
            card_pool = CardPool(cube)
            removed = CardPool(cube)

            # assessor = BasicAssessor(cube, card_pool, removed, color_num=color_num, alpha=alpha)
            # assessor = ParettoFrontAssessor(cube, card_pool, removed, color_num=color_num, alpha=alpha)
            # assessor = FixedEffectsAssessor(cube, card_pool, removed, color_num=color_num, alpha=alpha)
            assessor = FixedEffectsParettoFrontAssessor(cube, card_pool, removed, color_num=color_num, alpha=alpha)
            # assessor = SimpleMetricEmbeddingAssessor(cube, card_pool, removed)

            env = Environment(4, 3, 15, cube, card_pool, removed, assessor, draft_num=5)

            try:
                while True:
                    env.simulate_pick()
            except StopIteration:
                print(f"    {color_num} Done")

            arr = []
            for index, k in enumerate(card_pool):
                # print(index+2, end=". ")
                if k==env.bot_pool[index]:
                    if by_hand:
                        print(1)
                    else:
                        arr.append(1)
                else:
                    if by_hand:
                        print(0)
                    else:
                        arr.append(0)
            df_1.update({color_num: arr})

        df_1 = pd.DataFrame.from_dict(df_1)
        df_1.to_excel(f"./evaluation/results{alpha}.xlsx")

        df.update({alpha: df_1})
    if not by_hand:
        pass
        # df = pd.DataFrame.from_dict(df)
        # df.to_excel("./evaluation/basic_results.xlsx")
        # df.to_excel("./evaluation/pareto_results.xlsx")
        # df.to_excel("./evaluation/fe_results.xlsx")
        # df.to_excel("./evaluation/pareto_fe_results.xlsx")
        # df.to_excel("./evaluation/metric_results.xlsx")