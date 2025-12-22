import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import paretto_fronts


def smooth_winrate(alpha=0):
    def winrate(g: pd.DataFrame):
        return (g["wins"].sum() + alpha) / (g["wins"].sum() + g["losses"].sum() + 2*alpha)

    return winrate


def fe_model(df, alpha):
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
    return LinearRegression().fit(X, y)


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
    alpha = 0
    model = fe_model(df, alpha)
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

    for index, k in enumerate(fronts[:15]):
        paretto_fronts.create_graph(card_stats, k, index, True)
    # paretto_fronts.create_graph(card_stats, fronts[0], front_num=0, label=True)
    plt.savefig("pareto_fronts.png")
    plt.show()

    # X2 = sm.add_constant(X)
    # est = sm.OLS(y.T[0], X2).fit()
    # print(est.summary())
    # pass

    # params = np.append(model.intercept_, model.coef_)
    # predictions = model.predict(X)
    #
    # newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    # MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))
    #
    # # Note if you don't want to use a DataFrame replace the two lines above with
    # # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    #
    # matrix = np.dot(newX.T, newX)
    # inv = np.linalg.pinv(matrix)
    # var_b = MSE * inv.diagonal()
    # sd_b = np.sqrt(var_b)
    # ts_b = params / sd_b
    #
    # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
    #
    # sd_b = np.round(sd_b, 3)
    # ts_b = np.round(ts_b, 3)
    # p_values = np.round(p_values, 3)
    # params = np.round(params, 4)
    #
    # myDF3 = pd.DataFrame()
    # myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
    #                                                                                               p_values]
    # print(myDF3)
