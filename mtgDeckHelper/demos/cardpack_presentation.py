import random
from shutil import copyfileobj

import pandas as pd

from display_classes import CardDataFrame

import time
import tkinter as tk

from io import BytesIO

from PIL import Image, ImageTk
from requests import get
from json import loads


if __name__ == "__main__":
    cube = pd.read_csv('../alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})

    # Create the main window
    height = 1000
    width = 1300

    root = tk.Tk()
    geometry = str(width) + 'x' + str(height)
    root.geometry(geometry)

    col_size = 5
    row_size = 3
    images = []

    # numbers acquired through checking what the scryfall image resolution is
    img_height = 1040
    img_width = 745

    frame_height = int(height / row_size)
    frame_width = int(width / col_size)

    new_height = frame_height
    new_width = int(img_width * (frame_height / img_height))

    cardpack_frame = tk.Frame(root)
    cardpack_frame.pack(fill=tk.BOTH, expand=True)
    for k in range(col_size):
        cardpack_frame.columnconfigure(k, weight=1)
    for k in range(row_size):
        cardpack_frame.rowconfigure(k, weight=1)

    save = True

    selected = []
    for k in range(col_size):
        images.append([])
        for i in range(row_size):
            index = random.randint(0, len(cube) - 1)
            while index in selected:
                index = random.randint(0, len(cube) - 1)
            cardname = cube['card'][index]
            selected.append(index)

            img = None
            try:
                img = Image.open(f'card_images/{cardname.replace(" ", "_").replace("//", "")}.png')
                # img = mpimg.imread(f'card_images/{cardname.replace(" ", "_")}.png')
                # img = Image.fromarray(img)
            except FileNotFoundError as e:
                card = loads(get(f"https://api.scryfall.com/cards/search?q={cardname}").text)
                img_url = card['data']
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

                # mpimg.imsave(f'card_images/{cardname.replace(" ", "_")}.png', img)
                if save:
                    with open(f'../card_images/{cardname.replace(" ", "_").replace("//", "")}.png', 'wb') as out_file:
                        copyfileobj(get(img_url, stream=True).raw, out_file)

            img = img.resize((new_height, new_width), Image.Resampling.LANCZOS)

            img = ImageTk.PhotoImage(img)
            images[k].append(img)

            frame = CardDataFrame(cardpack_frame, img, cardname)

            frame.grid(row=i, column=k)

    # Start the GUI event loop
    root.mainloop()
