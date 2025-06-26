from tkinter import ttk

from card_overview import CardDataFrame
from display_classes import get_card_image


class Pack(ttk.Frame):
    def __init__(self, parent, starting_num_cards, card_pool, removed_cards, assessor, col_size=5, row_size=3, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.starting_num_cards = starting_num_cards
        self.assessor = assessor

        self.full = False
        self.removing = False

        self.col_size = col_size
        self.row_size = row_size
        self.card_pool = card_pool
        self.removed_cards = removed_cards

        self.cards = []     # frames to represent the cards
        self.cardnames = [] # names of all the cards in the pack
        self.images = []    # images of all the cards; required to be saved somewhere, otherwise they won't appear

        self.index = 0

        for k in range(col_size):
            self.columnconfigure(k, weight=1)
        for k in range(row_size):
            self.rowconfigure(k, weight=1)
            self.cards.append([])
            self.cardnames.append([])
            self.images.append([])

        self.listeners = []

    def is_full(self):
        return self.full

    def is_empty(self):
        return self.index == 0

    def is_removing(self):
        return self.removing

    def toggle_removing(self):
        self.removing = not self.removing
        for k in self.cards:
            for i in k:
                i.toggle_button_text()

    def add_card(self, cardname):
        """adds a card into the card pack"""
        if self.full:
            return

        img = get_card_image(cardname)
        if img is None:
            return

        row = self.index // self.col_size
        col = self.index - row * self.col_size

        self.cardnames[row].append(cardname)
        self.images[row].append(img)

        self.index += 1

        if self.index == self.starting_num_cards:
            self.full = True

        frame = CardDataFrame(self, img, cardname, self.assessor)
        frame.add_listener(lambda: self.remove_card(row, col))
        for k in self.listeners:
            frame.add_listener(k)

        frame.grid(row=row, column=col)
        self.cards[row].append(frame)

    def __len__(self):
        return self.index

    def get_data(self):
        """Return information on which cards are currently present in the pack"""
        return self.cardnames

    def remove_card(self, row, col):
        """removes a card from the card pack, either to go into the card pool or another player's card pool (removed)"""
        if not self.full or self.index == 0:
            return

        self.cards[row][col].grid_remove()
        self.index -= 1

        if self.is_removing():
            self.removed_cards.append(self.cardnames[row][col])
            # if self.is_empty():
        else:
            self.card_pool.append(self.cardnames[row][col])

        self.cardnames[row][col] = None

    def add_listener_to_all_buttons(self, func):
        self.listeners.append(func)
        for k in self.cards:
            for i in k:
                i.add_listener(func)

    def get_card(self, index):
        row = index // self.col_size
        col = index - row * self.col_size

        return self.cards[row][col]


class FilledPack(Pack):
    def __init__(self, parent, cards, card_pool, removed_cards, assessor, col_size=5, row_size=3, *args, **kwargs):
        super().__init__(parent, len(cards), card_pool, removed_cards, assessor, col_size, row_size, *args, **kwargs)

        for k in cards:
            self.add_card(k)