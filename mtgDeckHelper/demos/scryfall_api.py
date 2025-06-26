from io import BytesIO

from PIL import Image
from requests import get
from json import loads
from shutil import copyfileobj
from matplotlib import pyplot as plt

# In this example, we're looking for "Vindicate"
search_query = 'Counterspell'

# Load the card data from Scryfall
# card = loads(get(f"https://api.scryfall.com/cards/search?q={search_query}").text)
card = loads(get(f"https://api.scryfall.com/cards/random").text)


# Get the image URL
# img_url = card['data'][0]['image_uris']['png']
img_url = card['image_uris']['png']
print(img_url)

response = get(img_url)
img = Image.open(BytesIO(response.content))
plt.imshow(img)
plt.show()
# Save the image
# plt.imshow(get(img_url).raw)
# with open(f'card_images/{search_query}.png', 'wb') as out_file:
#    copyfileobj(get(img_url, stream=True).raw, out_file)
