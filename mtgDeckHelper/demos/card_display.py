import time
import tkinter as tk

from io import BytesIO

from PIL import Image, ImageTk
from requests import get
from json import loads

from display_classes import CardDataFrame

if __name__ == "__main__":
    # Create the main window
    height = 1000
    width = 750

    root = tk.Tk()
    geometry = str(width) + 'x' + str(height)
    root.geometry(geometry)

    col_size = 5
    row_size = 3
    images = []

    # frame = ttk.Frame(root)
    for k in range(col_size):
        root.columnconfigure(k, weight=1)
    for k in range(row_size):
        root.rowconfigure(k, weight=1)
    # frame.pack(fill=tk.BOTH, expand=True)

    for k in range(col_size):
        images.append([])
        for i in range(row_size):
            card = loads(get(f"https://api.scryfall.com/cards/random").text)
            img_url = card['image_uris']['png']
            response = get(img_url)
            img = Image.open(BytesIO(response.content))
            ratio = max(img.width / (width // row_size), img.height / (height // row_size))
            img = img.resize((int(img.height / ratio), int(img.width / ratio)), Image.Resampling.LANCZOS)

            img = ImageTk.PhotoImage(img)
            images[k].append(img)
            name = card['name']

            frame = CardDataFrame(root, img, name)

            frame.grid(row=i, column=k)

            # canvas.create_image(0, 0, anchor=NW, image=img)
            # username_label = ttk.Label(frame, text="CARD " + str(k*5+i + 1))
            # username_label.grid(column=k, row=i)

            time.sleep(0.1)

    # Start the GUI event loop
    root.mainloop()
