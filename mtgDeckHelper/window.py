import tkinter as tk
from display_classes import CardSelector
from pack import Pack, FilledPack


def switch_pack(window):
    # if pack is still not filled, do nothing
    if not window.packs[window.selected_pack].is_full():
        return

    if window.packs[window.selected_pack].is_removing():
        window.num_to_remove -= 1
        if window.num_to_remove == 0:
            window.packs[window.selected_pack].toggle_removing()
            window.num_to_remove = window.num_players - 1
            if window.packs[window.selected_pack].is_empty():
                window.reset_round()
        return

    window.packs[window.selected_pack].pack_forget()
    window.selected_pack = (window.selected_pack + 1) % len(window.packs)

    # if new pack is empty, round is done
    if window.packs[window.selected_pack].is_empty() and window.packs[window.selected_pack].is_full():
        window.reset_round()
        return

    window.packs[window.selected_pack].pack()
    if window.packs[window.selected_pack].is_full() and window.num_to_remove > 0 and (len(window.card_pool)%15>=window.num_players or len(window.card_pool)%15==0):
        window.packs[window.selected_pack].toggle_removing()
        # self.packs[self.selected_pack].update_scores(self.assessor)


class Window(tk.Tk):
    def __init__(self, available_cards, num_players, card_pool, removed_cards, assessor, singleton=False, num_rounds=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_rounds = num_rounds
        self.singleton = singleton
        self.assessor = assessor  # the thing that decides card scores

        self.available_cards = available_cards

        self.num_to_remove = num_players-1    # to count how many cards need to be removed before adding a new card to the card pool

        self.num_players = num_players

        self.card_pool = card_pool
        self.removed_cards = removed_cards

        self.packs = [Pack(self, 15-k, card_pool, removed_cards, self.assessor) for k in range(num_players)]
        self.add_button_listener(lambda: switch_pack(self))

        self.selected_pack = 0

        self.card_add_frame = CardSelector(self, available_cards)
        self.card_add_frame.add_button_listener(lambda: self.packs[self.selected_pack].add_card(self.card_add_frame.get()))
        # self.card_add_frame.add_button_listener(lambda: self.packs[self.selected_pack].get_card(len(self.packs[self.selected_pack])-1).set_card_score(assessor.calculate_card_score(self.card_add_frame.get())))
        if singleton:
            self.card_add_frame.listeners.append(lambda: self.card_add_frame.remove_card(self.card_add_frame.get()))
        self.card_add_frame.pack()

        self.packs[self.selected_pack].pack()

    def add_button_listener(self, func):
        """Add a listener to each button in the card pack"""
        for k in self.packs:
            k.add_listener_to_all_buttons(func)

    def get_pack_data(self):
        """Return data about all the cards currently present in all the card packs"""
        pack_data = []
        for pack in self.packs:
            pack_data.append(pack.get_data())

        return pack_data

    def reset_round(self):
        self.destroy()
        if self.num_rounds == 1:
            return

        new_window = Window(self.available_cards, self.num_players, self.card_pool, self.removed_cards, self.assessor, self.singleton, self.num_rounds-1)
        new_window.mainloop()


class PreconstructedWindow(tk.Tk):
    def __init__(self, order_cards: [], num_players, card_pool, removed_cards, assessor, num_rounds=3, listeners:[]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_rounds = num_rounds
        self.assessor = assessor  # the thing that decides card scores

        self.num_to_remove = num_players-1    # to count how many cards need to be removed before adding a new card to the card pool

        self.num_players = num_players

        self.card_pool = card_pool
        self.removed_cards = removed_cards

        self.order_cards = order_cards

        self.packs = [FilledPack(self, order_cards[k*15:(k+1)*15-k], card_pool, removed_cards, self.assessor) for k in range(num_players)]  # "-k" at the second part of the slide removes the last couple of cards from the pack
        self.selected_pack = 0

        self.packs[self.selected_pack].pack()

        self.listeners = []
        self.add_button_listener(lambda: switch_pack(self))
        # if listeners is not None:
        #     for listener in listeners:
        #         self.add_button_listener(listener)
        # else:
        #     self.add_button_listener(lambda: switch_pack(self))

    def add_button_listener(self, func):
        """Add a listener to each button in the card pack"""
        self.listeners.append(func)
        for k in self.packs:
            k.add_listener_to_all_buttons(func)

    def get_pack_data(self):
        """Return data about all the cards currently present in all the card packs"""
        pack_data = []
        for pack in self.packs:
            pack_data.append(pack.get_data())

        return pack_data

    def reset_round(self):
        self.destroy()
        if self.num_rounds == 1:
            return

        new_window = PreconstructedWindow(self.order_cards[15*self.num_players:], self.num_players, self.card_pool, self.removed_cards, self.assessor, self.num_rounds-1)
        for k in self.listeners[1:]:
            new_window.add_button_listener(k)
        new_window.mainloop()
