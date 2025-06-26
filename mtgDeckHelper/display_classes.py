import tkinter as tk
from shutil import copyfileobj
from tkinter import ttk

import time

from io import BytesIO

from PIL import Image, ImageTk
from requests import get
from json import loads

import autocomplete_combobox
from card_overview import CardDataFrame

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


def get_card_image(cardname, save=True):
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


class CardSelector(ttk.Frame):
    """Class has an inbuilt card list to select in order to construct a pack"""
    def __init__(self, parent, cards: list, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        label = ttk.Label(self, text="Please select a card:")
        label.grid(column=0, row=0, columnspan=2)
        selected_card = tk.StringVar(self)
        self.card_cb = autocomplete_combobox.AutocompleteCombobox(self, cards, textvariable=selected_card)
        # self.card_cb = ttk.Combobox(self, textvariable=selected_card)
        self.card_cb.grid(column=0, row=1, columnspan=2)
        # self.card_cb['values'] = sorted(cards)
        # self.card_cb['state'] = 'readonly'
        button = tk.Button(self, text="ADD", command=self.command)
        button.grid(column=1, row=1, columnspan=2)
        button.grid(column=1, row=1, columnspan=2)

        self.listeners = []

    def command(self):
        for k in self.listeners:
            k()

    def add_button_listener(self, func):
        self.listeners.append(func)

    def get(self):
        return self.card_cb.get()

    def remove_card(self, card):
        self.card_cb['values'].remove(card)
