import pandas as pd

import assessor_classes
from card_pool import CardPool
import queue

def load_data():
    cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    decks = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Decks',
                          usecols=['date', 'player', 'card'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})
    decks.date = decks.date.apply(lambda x: x.date())
    decks.set_index(['date', 'player'], inplace=True)

    games = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Games',
                          usecols=['date', 'player', 'opponent', 'wins', 'losses'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
                                 'losses': 'int'})
    games.date = games.date.apply(lambda x: x.date())
    games.set_index(['date', 'player'], inplace=True)

    return cube, decks, games

if __name__=="__main__":
    cube, decks, games = load_data()

    card_pool = CardPool(cube)
    removed = CardPool(cube)

    assessor = assessor_classes.SimpleMetricEmbeddingAssessor(cube, card_pool, removed)
    # assessor = assessor_classes.MetricEmbeddingAssessorClosest(cube, card_pool, removed)

    card = 'Abrade'
    card_pool.append(card)
    cube.drop(card,inplace=True)

    while len(card_pool)<45:
        pq = queue.PriorityQueue()
        for k in cube.index:
            score = assessor.calculate_card_score(k)
            pq.put((1-score,k))
        card = pq.get()[1]
        card_pool.append(card)
        print(card)
        cube.drop(card,inplace=True)
