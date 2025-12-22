import tkinter as tk
from tkinter import ttk


class AutocompleteCombobox(ttk.Combobox):
    """
    The textbox with an inbuilt list of names. Writing one will autocomplete it. Can also simply select an option from the list without writing it
    """
    def __init__(self, parent, completion_list, **kwargs):
        super().__init__(parent, **kwargs)
        self._completion_list = sorted(completion_list)
        # for i in range(len(self._completion_list)):
        #     self._completion_list[i] = self._completion_list[i].lower()
        self.position = 0
        self.configure(values=self._completion_list)
        self.bind('<KeyRelease>', self.handle_keyrelease)

    def autocomplete(self) -> None:
        """
        Check the list of names against the list of completions and complete it
        """
        curr_string = self.get().lower()

        # collect hits
        _hits = []
        for element in self._completion_list:
            if element.lower().startswith(curr_string):
                _hits.append(element)

        # if we have hits, do nothing if there are multiple; otherwise show the only hit
        if len(_hits) == 1:
            self.delete(0, tk.END)
            self.insert(0, _hits[0])
            self.select_range(len(curr_string), tk.END)
            self.icursor(len(curr_string))

    def handle_keyrelease(self, event) -> None:
        """
        listen for keypress events
        :param event: the event to decide which key was pressed
        """
        if event.keysym in ("BackSpace", "Left", "Right", "Up", "Down"):
            return
        self.autocomplete()


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Autocomplete Combobox")

    completion_list =  ['Apple', 'Banana', 'Blueberry', 'Cherry', 'Date', 'Grape', 'Orange', 'Peach', 'Pear', 'Plum']

    combo = AutocompleteCombobox(root, completion_list)

    combo.pack()

    root.mainloop()