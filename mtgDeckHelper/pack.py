from tkinter import ttk

from card_overview import CardDataFrame

import time

from io import BytesIO

from PIL import Image, ImageTk
from requests import get
from json import loads
from shutil import copyfileobj

#TODO move this hard-coded window specification into the window
height = 1000
width = 1300

col_size = 5
row_size = 3
images = []

# numbers acquired by checking the scryfall png image resolution
img_height = 1040
img_width = 745

frame_height = int(height / row_size)
frame_width = int(width / col_size)

new_height = frame_height
new_width = int(img_width * (frame_height / img_height))

def get_card_image(cardname:str, save:bool=True) -> ImageTk.PhotoImage:
    """
    Returns the image of the given card using the scryfall API. If card has multiple versions, returns the newest one
    :param cardname: Name of the card
    :param save: Whether to save the image (if image of that name doesn't already exist), default true
    :return: Tk PhotoImage
    """
    try:
        img = Image.open(f'card_images/{cardname.replace(" ", "_").replace("//", "")}.png')
    except FileNotFoundError:
        card = loads(get(f"https://api.scryfall.com/cards/search?q={cardname}").text)
        try:
            img_url = card['data']
        except KeyError:
            return None
        # in case there are more cards that contain the name, we need to find the correct one
        j = 0
        if len(img_url) > 1:
            while img_url[j]['name'] != cardname:
                j += 1
        img_url = img_url[j]
        try:
            img_url = img_url['image_uris']['png']
        except KeyError:
            # in case a card has 2 faces, we just need the front one
            img_url = img_url['card_faces'][0]['image_uris']['png']
        response = get(img_url)
        img = Image.open(BytesIO(response.content))
        time.sleep(0.1)
        if save:
            with open(f'./card_images/{cardname.replace(" ", "_").replace("//", "")}.png', 'wb') as out_file:
                 copyfileobj(get(img_url, stream=True).raw, out_file)

    img = img.resize((new_height, new_width), Image.Resampling.LANCZOS)

    return ImageTk.PhotoImage(img)

#TODO calculate the number of rows and columns per number of cards per pack. Can also be done by the window so as to not be done multiple times
# but it shouldn't make that much of a difference and the pack should control its layout anyways. Currently the column and row size are hard-coded,
# and that can become a problem for larger packs
class Pack(ttk.Frame):
    """
    Models a single pack of cards
    """
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
        self.listeners.append(self._set_best_card)
        self.pack_load_listeners = []
        self.pack_load_listeners.append(self._set_best_card)

    def is_full(self) -> bool:
        """

        :return: true if the pack is full (even if players have since started taking the cards from it), false otherwise
        """
        return self.full

    def is_empty(self) -> bool:
        """
        :return: true if the pack is empty, false otherwise
        """
        return self.index == 0

    def is_removing(self) -> bool:
        """
        :return: true if the other players are picking cards instead of the main player, false otherwise
        """
        return self.removing

    def toggle_removing(self) -> None:
        """
        change who is picking cards from the pack
        """
        self.removing = not self.removing
        for k in self.cards:
            for i in k:
                i.toggle_button_text()

    def listen_load(self) -> None:
        """
        activate listeners activated when the pack is loaded in
        """
        for k in self.pack_load_listeners:
            k()

    def add_card(self, cardname:str) -> None:
        """
        adds a card into the card pack
        :param cardname: the name of the card
        """
        if self.full:
            # self.listen_load()
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

        self.listen_load()

    def __len__(self):
        """
        :return: the number of cards in the pack
        """
        return self.index

    def get_data(self) -> list:
        """Return information on which cards are currently present in the pack"""
        return self.cardnames

    def remove_card(self, row:int, col:int) -> None:
        """
        removes a card from the card pack, either to go into the card pool or another player's card pool
        :param row: the index of the card row
        :param col: the index of the card column"""
        if not self.full or self.index == 0:
            return

        self.cards[row][col].grid_remove()
        # self.cards[row].pop(col)
        self.index -= 1

        if self.is_removing():
            self.removed_cards.append(self.cardnames[row][col])
            # if self.is_empty():
        else:
            self.card_pool.append(self.cardnames[row][col])

        self.cardnames[row][col] = None

    def add_listener_to_all_buttons(self, func:callable) -> None:
        """
        Adds a listener to the pack when any button is pressed
        :param func: the function to call upon a button press.
        """
        self.listeners.append(func)
        for k in self.cards:
            for i in k:
                i.add_listener(func)

    def add_pack_load_listener(self, func:callable) -> None:
        """
        Adds a listener to the pack when the pack is rendered
        :param func: the function to call upon pack render
        """
        self.pack_load_listeners.append(func)

    def get_card(self, index:int) -> str:
        """
        Breaks the index down into row and column coordinates and returns the card name at that position
        :param index: The index of the desired card
        :return: the card name at the given index
        """
        row = index // self.col_size
        col = index - row * self.col_size

        return self.cards[row][col]

    def _set_best_card(self) -> None:
        """
        Chooses the best card from the pack according to the assessor
        """
        if not self.is_full():
            return
        best_card = self.cards[0][0]
        best_score = self.assessor.calculate_card_score(best_card.get_cardname())
        for k in self.cards:
            for i in k:
                i.set_background_color("red")
                cardname = i.get_cardname()
                score = self.assessor.calculate_card_score(cardname)
                if (score > best_score and cardname not in self.removed_cards and cardname not in self.card_pool) or best_card.get_cardname() in self.removed_cards or best_card.get_cardname() in self.card_pool:
                    best_card = i
                    best_score = score
        best_card.set_background_color("green")


class FilledPack(Pack):
    """
    Pack that has already been filled with cards, based on the cards in a list
    """
    def __init__(self, parent, cards, card_pool, removed_cards, assessor, col_size=5, row_size=3, *args, **kwargs):
        super().__init__(parent, len(cards), card_pool, removed_cards, assessor, col_size, row_size, *args, **kwargs)

        for k in cards:
            self.add_card(k)