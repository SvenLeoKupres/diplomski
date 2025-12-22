from typing import Callable

import pandas as pd


class CardPool:
    """
    Models a pool of cards
    """
    def __init__(self, cube):
        """
        :param cube: a dataframe which lists all the cards that can be added to the pool
        """
        self.cards = []
        self.cube = cube

        self.listeners = []
        self.listeners.append(self._update_color)

        self.colors = {"W":0, "U":0, "B":0, "R":0, "G":0}

    def append(self, cardname):
        """
        Adds a new card to the pool
        :param cardname: name of the card
        """
        self.cards.append(cardname)
        for listener in self.listeners:
            listener(cardname, "append")

    def remove(self, cardname):
        """
        Removes a card from the pool
        :param cardname: name of the card
        """
        self.cards.remove(cardname)
        for listener in self.listeners:
            listener(cardname, "remove")

    def add_change_listener(self, listener: Callable[[str, str], None]) -> None:
        """
        Register a change listener - anything that triggers when a card is added to or removed from the pool
        :param listener: the listener being triggered. Has to be a function which takes in a card name and event type as inputs.
        """
        self.listeners.append(listener)

    def remove_change_listener(self, listener) -> None:
        """
        Remove a change listener
        :param listener: the listener being removed
        """
        self.listeners.remove(listener)

    def get_color_dict(self) -> dict:
        """

        :return: a dictionary where the keys are colors (including colorless), and the values are the number of cards of that color in the pool
        (multicolored cards are counted multiple times)
        """
        return self.colors

    def _update_color(self, cardname:str, event:str) -> None:
        """
        Updates the dictionary of colors after a card is added or removed
        """
        card_color = self.cube.loc[cardname, "color"]

        if type(card_color) is float:   # if card is colorless
            return

        if event == "append":
            for k in card_color:
                self.colors[k] += 1
        elif event == "remove":
            for k in card_color:
                self.colors[k] -= 1

    def __iter__(self):
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, index: int) -> str:
        return self.cards[index]

if __name__ == "__main__":
    cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    pool = CardPool(cube)
    pool.append("Abrade")
    pool.remove("Abrade")
    pass