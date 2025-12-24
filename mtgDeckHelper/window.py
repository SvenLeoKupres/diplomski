import tkinter as tk

from card_selector import CardSelector
from pack import Pack, FilledPack


def optimal_pack_dimensions(num_cards:int) -> (int, int):
    """
    Calculate the optimal number of rows and columns for the given number of cards per pack
    :param num_cards: number of cards in the pack
    :return: tuple, first number is the number of rows, second is the number of columns
    """
    # x * y = num_cards
    # x = y+C, minimizirati C, tj minimizirati x-y
    root = int(num_cards**0.5)
    proper = []
    for i in range(1, root+1):
        for j in range(i, num_cards+1):
            if i*j == num_cards:
                proper.append((i,j))

    k = 0
    best = k
    while k<len(proper):
        item = proper[k]
        if item[1] > item[0] + 4:
            proper.pop(k)
        elif item[1]-item[0]<proper[best][1]-proper[best][0]:
            proper.pop(best)
        else:
            k += 1

    if len(proper)==0:
        return optimal_pack_dimensions(num_cards+1)

    return proper[0]


class AbstractWindow(tk.Tk):
    """
    Basis for a main window. Supports no functionality by itself.
    """
    def __init__(self, num_players, card_pool, removed_cards, assessor, num_rounds=3, num_cards_per_pack=15, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_players = num_players
        self.card_pool = card_pool
        self.removed_cards = removed_cards
        self.assessor = assessor
        self.num_rounds = num_rounds
        self.num_cards_per_pack = num_cards_per_pack
        self.row_size, self.col_size = optimal_pack_dimensions(self.num_cards_per_pack)

        self.packs = []
        self.selected_pack = 0
        self.num_to_remove = num_players - 1    # to count how many cards need to be removed before adding a new card to the card pool

    def add_button_listener(self, func: callable) -> None:
        """
        Add a listener to each button in the card pack
        :param func: Function to call on each button. Requires no inputs and returns nothing"""
        for k in self.packs:
            k.add_listener_to_all_buttons(func)

    def get_pack_data(self) -> []:
        """
        :return: data about all the cards currently present in all the card packs
        """
        pack_data = []
        for pack in self.packs:
            pack_data.append(pack.get_data())

    def reset_round(self) -> None:
        """
        If there are more rounds to be played, creates a new environment to continue. Otherwise, destroys the window
        """
        self.destroy()
        if self.num_rounds == 1:
            return

    def switch_pack(self) -> None:
        """
        Switch pack and update any necessary visuals
        """
        # if pack is still not filled, do nothing
        if not self.packs[self.selected_pack].is_full():
            return

        if self.packs[self.selected_pack].is_removing():
            self.num_to_remove -= 1
            if self.num_to_remove == 0:
                self.packs[self.selected_pack].toggle_removing()
                self.num_to_remove = self.num_players - 1
                if self.packs[self.selected_pack].is_empty():
                    self.reset_round()
            return

        self.packs[self.selected_pack].pack_forget()
        self.selected_pack = (self.selected_pack + 1) % len(self.packs)

        # if new pack is empty, round is done
        if self.packs[self.selected_pack].is_empty() and self.packs[self.selected_pack].is_full():
            self.reset_round()
            return

        self.packs[self.selected_pack].pack()
        if self.packs[self.selected_pack].is_full() and self.num_to_remove > 0 and (
                len(self.card_pool) % self.num_cards_per_pack >= self.num_players or len(self.card_pool) % self.num_cards_per_pack == 0):
            self.packs[self.selected_pack].toggle_removing()
            # self.packs[self.selected_pack].update_scores(self.assessor)

        # set off listeners which track when there was an update in a pack
        self.packs[self.selected_pack].listen_load()


class ProperWindow(AbstractWindow):
    """
    Window to use when actually drafting during a game. User is able to manually add cards to the selected pack.
    """
    def __init__(self, available_cards, num_players, card_pool, removed_cards, assessor, singleton=False, num_rounds=3, num_cards_per_pack=15, *args, **kwargs):
        super().__init__(num_players, card_pool, removed_cards, assessor, num_rounds, num_cards_per_pack, *args, **kwargs)

        self.singleton = singleton
        self.available_cards = available_cards

        self.packs = [Pack(self, num_cards_per_pack-k, card_pool, removed_cards, self.assessor, col_size=self.col_size, row_size=self.row_size) for k in range(num_players)]
        self.add_button_listener(lambda: self.switch_pack())

        self.selected_pack = 0

        self.card_add_frame = CardSelector(self, available_cards)
        self.card_add_frame.add_button_listener(lambda: self.packs[self.selected_pack].add_card(self.card_add_frame.get()))
        # self.card_add_frame.add_button_listener(lambda: self.packs[self.selected_pack].get_card(len(self.packs[self.selected_pack])-1).set_card_score(assessor.calculate_card_score(self.card_add_frame.get())))
        if singleton:
            self.card_add_frame.listeners.append(lambda: self.card_add_frame.remove_card(self.card_add_frame.get()))
        self.card_add_frame.pack()

        self.packs[self.selected_pack].pack()

        self.packs[self.selected_pack].listen_load()

    def add_button_listener(self, func):
        for k in self.packs:
            k.add_listener_to_all_buttons(func)

    def get_pack_data(self):
        pack_data = []
        for pack in self.packs:
            pack_data.append(pack.get_data())

        return pack_data

    def reset_round(self):
        self.destroy()
        if self.num_rounds == 1:
            return

        new_window = ProperWindow(self.available_cards, self.num_players, self.card_pool, self.removed_cards, self.assessor, self.singleton, self.num_rounds - 1, self.num_cards_per_pack)
        new_window.mainloop()


class PreconstructedWindow(AbstractWindow):
    """
    User must give a preselected card order for the packs. The length of the list must be equal to num_players*num_rounds*num_cards_per_pack.
    """
    def __init__(self, order_cards: [], num_players, card_pool, removed_cards, assessor, num_rounds=3, num_cards_per_pack=15, listeners:[]=None, *args, **kwargs):
        super().__init__(num_players, card_pool, removed_cards, assessor, num_rounds, num_cards_per_pack, *args, **kwargs)

        if len(order_cards) != num_players*num_rounds*num_cards_per_pack:
            raise ValueError("inappropriate length of list")

        self.order_cards = order_cards

        self.packs = [FilledPack(self, order_cards[k*num_cards_per_pack:(k+1)*num_cards_per_pack-k], card_pool, removed_cards, self.assessor, row_size=self.row_size, col_size=self.col_size) for k in range(num_players)]  # "-k" at the second part of the slide removes the last couple of cards from the pack
        self.selected_pack = 0

        self.packs[self.selected_pack].pack()

        self.listeners = []
        self.add_button_listener(lambda: self.switch_pack())
        # if listeners is not None:
        #     for listener in listeners:
        #         self.add_button_listener(listener)
        # else:
        #     self.add_button_listener(lambda: switch_pack(self))

        self.packs[self.selected_pack].listen_load()

    def add_button_listener(self, func):
        self.listeners.append(func)
        for k in self.packs:
            k.add_listener_to_all_buttons(func)

    def get_pack_data(self):
        pack_data = []
        for pack in self.packs:
            pack_data.append(pack.get_data())

        return pack_data

    def reset_round(self):
        self.destroy()
        if self.num_rounds == 1:
            return

        new_window = PreconstructedWindow(self.order_cards[self.num_cards_per_pack*self.num_players:], self.num_players, self.card_pool, self.removed_cards, self.assessor, self.num_rounds-1, self.num_cards_per_pack)
        for k in self.listeners[1:]:
            new_window.add_button_listener(k)
        new_window.mainloop()


if __name__=="__main__":
    for k in range(1, 21):
        print(f"{k}: {optimal_pack_dimensions(k)}")