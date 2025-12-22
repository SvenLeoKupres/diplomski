import pandas as pd

import paretto_fronts
from assessor_classes import smooth_winrate, extract_deck_count

def calculate_card_points(fronts, cardname):
    score = len(fronts)

    k = 0
    while k < score and cardname not in fronts[k]:
        k += 1
    return score - k

def calculate_deck_score(fronts, cards):
    """Calculates card score based on the assessor data about the cardname's Paretto front and how many colors in the picked pool it shares a color with"""
    total = 0
    for card in cards["card"]:
        total += calculate_card_points(fronts, card)

    return total

if __name__=="__main__":
    cube = pd.read_csv('../alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    decks = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Decks',
                          usecols=['date', 'player', 'card'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})

    decks.date = decks.date.apply(lambda x: x.date())
    decks.set_index(['date', 'player'], inplace=True)

    games = pd.read_excel(io='../alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Games',
                          usecols=['date', 'player', 'opponent', 'wins', 'losses'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
                                 'losses': 'int'})
    games.date = games.date.apply(lambda x: x.date())
    games.set_index(['date', 'player'], inplace=True)

    alpha = 0

    df = decks.merge(games, on=["date", "player"])

    card_wr = df.groupby(["card"]).apply(smooth_winrate(alpha), include_groups=False).rename("winrate")

    cardnames = decks.loc[:, "card"].sort_values().unique()
    not_played = cube[~cube.index.isin(cardnames)].loc[:, []]
    card_wr = pd.concat([card_wr, not_played], axis=1)
    card_wr.fillna(value=min(0.5, alpha), inplace=True)

    count = extract_deck_count(decks)
    count = pd.concat([count, not_played], axis=1)
    count.fillna(value=0, inplace=True)

    card_stats = pd.concat([count, card_wr], axis=1)

    fronts = paretto_fronts.divide_into_fronts(card_stats)

    cube["points"] = [calculate_card_points(fronts, k) for k in cube.index]

    combined_df = pd.merge(cube, decks.reset_index(), on="card")

    deck_points = combined_df.groupby(["date", "player"]).apply(lambda cards: sum(cards["points"])).rename("total_score").to_frame()
    deck_points["winrate"] = df.groupby(['date', 'player']).apply(lambda x: sum(x['wins'])/(sum(x['wins']) + sum(x['losses'])))

    pass
