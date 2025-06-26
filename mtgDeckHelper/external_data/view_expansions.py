import time

import pandas as pd
from json import loads
from requests import get

def get_card(cardname):
    card = loads(get(f"https://api.scryfall.com/cards/search?q={cardname}").text)['data']
    j = 0
    bull = False
    if len(card) > 1:
        while card[j]['name'] != cardname:
            j += 1
            if j == len(card) and not bull:
                bull = True
                j = 0
            if bull:
                card[j]['name'] = card[j]['name'].split(" // ")[0]
    card = card[j]

    # prints = loads(get(card["set_search_uri"]).text)
    prints_2 = loads(get(card['prints_search_uri']).text)['data']
    sets = []
    for k in prints_2:
        set_code = k['set']
        sets.append(set_code)
    sets = list(set(sets))

    return sets

if __name__=='__main__':
    # get_card("Harrow")
    # get_card("Experiment Kraj")
    cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'Set'],
                       dtype={'name': 'str', 'Set': 'str'})
    cube = cube.rename(columns={"name": "card", "Set": "set"})
    cube.set_index('card', inplace=True)

    cube.sort_values(by='set', inplace=True)

    expansion_per_card = dict()
    for k in cube.index:
        tmp = get_card(k)
        expansion_per_card[k] = tmp
        time.sleep(0.1)

    rows = [(key, val) for key, values in expansion_per_card.items() for val in values]
    expansions = pd.DataFrame(rows, columns=['card', 'set'])
    expansions.set_index('card', inplace=True)
    expansions = pd.pivot_table(expansions, index=expansions.index, columns='set', fill_value=0, aggfunc=lambda x: 1)
    expansions.to_csv('expansions.csv')

    pass