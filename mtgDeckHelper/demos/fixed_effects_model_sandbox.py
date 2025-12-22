import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import paretto_fronts
from demos.fixed_effects_model import fe_model


def smooth_winrate(alpha=0):
    def winrate(g: pd.DataFrame):
        return (g["wins"].sum() + alpha) / (g["wins"].sum() + g["losses"].sum() + 2*alpha)

    return winrate


if __name__=='__main__':
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

    df = decks.merge(games, on=["date", "player"])

    decks_t = pd.pivot_table(
        decks,
        index=decks.index,  # since the index already includes 'date' and 'player'
        columns='card',
        aggfunc=lambda x: 1,  # set to 1 where card was used
        fill_value=0
    )

    decks_t.columns.name = None
    decks_t.columns = [f'card_{col}' for col in decks_t.columns]

    decks_t.index = pd.MultiIndex.from_tuples(decks_t.index, names=['date', 'player'])
    decks_t = decks_t.reset_index()
    decks_t.set_index(['date', 'player'], inplace=True)

    alpha = 0  # factor that determines smoothing
    player_wr = df.groupby("player").apply(smooth_winrate(alpha)).rename("winrate")
    daily_player_wr = df.groupby(["date", "player"]).apply(smooth_winrate(alpha)).rename("daily_winrate").to_frame()
    daily_player_wr['winrate'] = daily_player_wr.index.get_level_values('player').map(player_wr)
    daily_player_wr["lift"] = daily_player_wr["daily_winrate"] - daily_player_wr["winrate"]

    card_wr = df.groupby("card").apply(smooth_winrate(alpha)).rename("winrate")
    daily_card_wr = df.groupby(["date", "card"]).apply(smooth_winrate(alpha)).rename("daily_winrate").to_frame()
    daily_card_wr["winrate"] = daily_card_wr.index.get_level_values('card').map(card_wr)
    daily_card_wr["lift"] = daily_card_wr["daily_winrate"] - daily_card_wr["winrate"]


    y = np.nan_to_num(daily_player_wr.loc[:, "lift"].unstack().to_numpy(), nan=0)
    X = np.nan_to_num(daily_card_wr.loc[:, "lift"].unstack().to_numpy(), nan=0)

    # y = y.sum(axis=0).reshape(-1, 1)
    # X = X.sum(axis=0).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    # print(model.score(X, y))
    # print(model.coef_)
    # print(model.intercept_)

    diff = model.coef_.sum(axis=0)
    # print(diff.shape)
    # print(diff)
    # true_winrate = 0.5 + diff
    # print(true_winrate)

    cardnames = decks.loc[:, "card"].sort_values().unique()
    # print(cardnames)

    card_lift = pd.Series(data=diff, index=cardnames).rename("winrate")
    # print(card_lift.head())

    game_count = df.groupby(["card"]).apply(
        lambda g: g["wins"].sum() + g["losses"].sum()
    ).rename("game_count")
    card_stats = pd.concat([game_count, card_lift], axis=1)

    fronts = paretto_fronts.divide_into_fronts(card_stats)

    for index, k in enumerate(fronts[:5]):
        paretto_fronts.create_graph(card_stats, k, index, True)
    # paretto_fronts.create_graph(card_stats, fronts[0], front_num=0, label=True)
    plt.show()

    X2 = sm.add_constant(X)
    est = sm.OLS(y.T[0], X2).fit()
    print(est.summary())
    pass