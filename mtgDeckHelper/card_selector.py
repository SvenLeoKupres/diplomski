import tkinter as tk
from tkinter import ttk

import autocomplete_combobox


class CardSelector(ttk.Frame):
    """Class has an inbuilt card list to select from in order to construct a pack"""
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
        button = tk.Button(self, text="ADD", command=self._command)
        button.grid(column=1, row=1, columnspan=2)
        button.grid(column=1, row=1, columnspan=2)

        self.listeners = []

    def _command(self) -> None:
        """Activate listeners of this object. To be used only upon activating the button"""
        for k in self.listeners:
            k()

    def add_button_listener(self, func: callable):
        """
        Append a functionality to the button
        :param func: Function to call
        """
        self.listeners.append(func)

    def remove_card(self, card: str) -> None:
        """
        Remove a card from the list
        :param card: name of the card to remove
        """
        self.card_cb['values'].remove(card)

    def get(self) -> str:
        """
        :return: the string currently written in the text bar
        """
        return self.card_cb.get()
