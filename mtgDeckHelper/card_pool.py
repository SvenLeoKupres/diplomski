class CardPool:
    def __init__(self, cube):
        self.cards = []
        self.cube = cube

        self.listeners = []
        self.listeners.append(self._update_color)

        self.colors = {"W":0, "U":0, "B":0, "R":0, "G":0}

    def append(self, cardname):
        self.cards.append(cardname)
        for listener in self.listeners:
            listener()

    def remove(self, cardname):
        self.cards.remove(cardname)
        for listener in self.listeners:
            listener()

    def add_change_listener(self, listener):
        self.listeners.append(listener)

    def get_color_dict(self):
        return self.colors

    def _update_color(self, ):
        cardname = self.cards[-1]
        card_color = self.cube.loc[cardname, "color"]

        if type(card_color) is float:
            return

        for k in card_color:
            self.colors[k] += 1

    def __iter__(self):
        return iter(self.cards)

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, index):
        return self.cards[index]