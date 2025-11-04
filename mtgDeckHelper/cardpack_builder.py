import random
import tkinter as tk
from tkinter import ttk

import pandas as pd

from card_pool import CardPool
# from evaluation.assess_metric import MetricEmbeddingAssessor
from window import Window, PreconstructedWindow
from assessor_classes import BasicAssessor, FixedEffectsAssessor, SimpleMetricEmbeddingAssessor, ParettoFrontAssessor, \
    FixedEffectsParettoFrontAssessor, CompositeAssessor, AbstractAssessor


class TextBuilder:
    def __init__(self, num_players, num_rounds, card_pool, removed, num_cards_per_pack=15):
        self.num_players = num_players
        self.num_rounds = num_rounds
        self.card_pool = card_pool
        self.removed = removed
        self.num_cards_per_pack = num_cards_per_pack
        self.card_pool.add_change_listener(self.update_card_pool)
        self.removed.add_change_listener(self.update_removed)
        self.text = f"num_players: {num_players}\nnum_rounds: {num_rounds}\nnum_cards_per_pack: {num_cards_per_pack}\n"

    def update_card_pool(self):
        self.text += f"added: {self.card_pool.cards[-1]}\nremoved: "
        # if len(self.card_pool.cards) == self.num_rounds*15:
        #     self.save()

    def update_removed(self):
        self.text += f"{self.removed.cards[-1]}; "
        n = self.num_players
        if len(self.removed.cards) == self.num_rounds*self.num_cards_per_pack*(n-1) - (self.num_rounds * ((n-2)*(n-1)+n*(n-1)) // 2):
            self.save()

    def save(self):
        with open("./draft", 'w') as f:
            f.write(self.text)


def save(packs, num_players, num_rounds, num_cards_per_pack, path='./packs'):
    data = f"{num_players};{num_rounds};{num_cards_per_pack};{packs[0]}"
    for index in range(1, len(packs)):
        data += f";{packs[index]}"
    with open(path, 'w') as f:
        f.write(data)

def load(path = './packs'):
    try:
        with open(path, 'r') as f:
            packs = f.read()
            data = packs.split(";")
            return int(data[0]), int(data[1]), int(data[2]), data[3:]
    except FileNotFoundError:
        raise FileNotFoundError("File doesn't exist")


def real_main(start_root, num_players_spinbox, rounds_spinbox, cards_spinbox, mode_spinbox):
    num_players = int(num_players_spinbox.get())
    num_rounds = int(rounds_spinbox.get())
    num_cards = int(cards_spinbox.get())
    mode = mode_spinbox.get()

    start_root.destroy()

    cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    card_pool = CardPool(cube) # pool of selected cards
    removed_cards = CardPool(cube) # pool of cards selected by other players

    # assessor = AbstractAssessor(cube, card_pool, removed_cards)
    # assessor_fixed_pareto = FixedEffectsParettoFrontAssessor(cube, card_pool, removed_cards)
    # assessor = FixedEffectsAssessor(cube, card_pool, removed_cards)
    # assessor = ParettoFrontAssessor(cube, card_pool, removed_cards)
    # assessor1 = BasicAssessor(cube, card_pool, removed_cards, color_num=1)
    # assessor3 = BasicAssessor(cube, card_pool, removed_cards, color_num=3)
    # assessor = CompositeAssessor(cube, card_pool, removed_cards, {"color_num=1": assessor1, "color_num=3": assessor3})
    # assessor5 = BasicAssessor(cube, card_pool, removed_cards, color_num=5)
    # assessor10 = BasicAssessor(cube, card_pool, removed_cards, color_num=10)
    # assessor = CompositeAssessor(cube, card_pool, removed_cards, {"color_num=5": assessor5, "color_num=10": assessor10})
    # assessor0 = BasicAssessor(cube, card_pool, removed_cards, color_num=0)
    # assessor20 = BasicAssessor(cube, card_pool, removed_cards, color_num=20)
    # assessor = CompositeAssessor(cube, card_pool, removed_cards, {"color_num=0": assessor0, "color_num=20": assessor20})
    # assessor = CompositeAssessor(cube, card_pool, removed_cards, {"metric embedding":assessor_metric, "fixed effects pareto":assessor_fixed_pareto, "basic":assessor5})
    assessor = SimpleMetricEmbeddingAssessor(cube, card_pool, removed_cards)

    # create the main window
    if mode=="game":
        root = Window(cube.index.tolist(), num_players, card_pool, removed_cards, assessor, False, num_rounds)
    elif mode=="random":
        shuffled_packs = cube.index.tolist()
        shuffled_packs = random.sample(shuffled_packs, num_cards*num_players*num_rounds)
        # save(shuffled_packs, num_players, num_rounds, num_cards)
        root = PreconstructedWindow(shuffled_packs, num_players, card_pool, removed_cards, assessor, num_rounds, num_cards)
    elif mode=="load file":
        num_players, num_rounds, num_cards, packs = load()
        root = PreconstructedWindow(packs, num_players, card_pool, removed_cards, assessor, num_rounds, num_cards)
    else:
        raise ValueError("Invalid game mode")
    # text_builder = TextBuilder(num_players, num_rounds, card_pool, removed_cards)

    root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    root.title('Game setup')

    title = tk.Label(root, text='Number of players')
    title.pack()
    player_spinbox = ttk.Spinbox(root, from_=0, to=10)
    player_spinbox.pack()

    title = tk.Label(root, text='Number of rounds')
    title.pack()
    rounds_spinbox = ttk.Spinbox(root, from_=0, to=10)
    rounds_spinbox.pack()

    title = tk.Label(root, text='Number of cards per pack')
    title.pack()
    cards_spinbox = ttk.Spinbox(root, from_=0, to=25)
    cards_spinbox.pack()

    title = tk.Label(root, text='Game mode')
    title.pack()
    mode_spinbox = ttk.Spinbox(root)
    mode_spinbox.configure(values=["game", "random", "load file"], state="readonly")
    mode_spinbox.pack(pady=20)

    title = tk.Label(root, text='(if mode is \"load file\", first 3 values are ignored)')
    title.pack()

    button = ttk.Button(root, text="Cube", command=lambda: real_main(root, player_spinbox, rounds_spinbox, cards_spinbox, mode_spinbox))
    button.pack()

    root.mainloop()
