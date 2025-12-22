import tkinter as tk
from tkinter import ttk

import assessor_classes


# part of the code from https://blog.finxter.com/5-best-ways-to-display-a-message-when-hovering-over-something-with-mouse-cursor-in-tkinter-python/
class CardInformation:
    """
    A popup that displays assessor information about a card
    """
    def __init__(self, widget, cardname:str, assessor:assessor_classes.AbstractAssessor):
        """

        :param widget: Above what to hover to display infomation
        :param cardname: name of the card being assessed
        :param assessor: assessor from which information is being gathered
        """
        self.widget = widget
        self.cardname = cardname
        self.assessor = assessor

        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tooltip_window = None

    def enter(self, event=None) -> None:
        """
        Displays infomation
        :param event: not relevant for the function of this method. Used to learn from whence the activation came
        """
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry("+%d+%d" % (x, y))

        text = self.cardname + "\n\n"
        #
        # score = self.assessor.calculate_card_score(self.cardname)
        # text += f"Average distance from deck: {score[0]:.4f}\n\n"
        #
        # text += "Best card synergies:\n"
        # card_distances = score[1]
        # for k in card_distances.keys():
        #     text += f"{k}, distance: {card_distances[k]: .4f}\n"

        label = tk.Label(self.tooltip_window, text=text + self.assessor.result_string(self.cardname), justify='left',
                         background='#ffffff', relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)
    def leave(self, event=None) -> None:
        """
        Removes the information from the screen
        :param event: not relevant for the function of this method. Used to learn from whence the activation came
        """
        if self.tooltip_window:
            self.tooltip_window.destroy()


class CardDataFrame(ttk.Frame):
    """
    A part of a pack. Displays card image, card name, and incorporates a button which allows the card to be picked as part of the draft
    """
    def __init__(self, parent, img, cardname:str, assessor:assessor_classes.AbstractAssessor, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Label(self, image=img)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # label = tk.Label(self, text=cardname, font=("Times New Roman", 25))
        # label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # self.label = tk.Label(self, font=("Times New Roman", 25))
        # self.label.pack()

        self.button_text = "PICK"
        self.button = tk.Button(self, text=self.button_text, font=("Times New Roman", 25), command=self._command)
        self.button.pack()

        self.listeners = []

        self.card_information = CardInformation(self, cardname, assessor)

    def _command(self):
        for k in self.listeners:
            k()

    def add_listener(self, func:callable) -> None:
        """

        :param func: function to activate upon button press. Must be callable
        """
        self.listeners.append(func)

    def toggle_button_text(self) -> None:
        """
        Toggles button text between "PICK" and "REMOVED" based on the pack mode
        """

        if self.button_text == "PICK":
            self.button_text = "REMOVE"
            self.button.configure(text="REMOVE")
        else:
            self.button_text = "PICK"
            self.button.configure(text="PICK")

    def get_cardname(self) -> str:
        """

        :return: name of the card this frame displays information on
        """
        return self.card_information.cardname

    def set_background_color(self, color) -> None:
        """
        Change the background color of the button. Used mainly to easily identify which card is the best pick from the pack
        :param color: Color to change the button to
        """
        self.button.config(bg=color)

    # def set_card_score(self, num):
    #     self.label.config(text=num)


if __name__ == '__main__':
    root = tk.Tk()
    button = tk.Button(root, text="Hover me")
    button.pack()
    tooltip = CardInformation(button, "Abrade", assessor_classes.SimpleMetricEmbeddingAssessor(None, None, None))
    root.mainloop()