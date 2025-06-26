import itertools

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import pandasql as ps


def divide_into_fronts(card_winrates):
    card_doms = dict()
    card_names = card_winrates.index

    fronts = []

    for k in card_names:
        card_doms[k] = 0

    combinations = list(itertools.product(card_names, card_names))
    for combination in combinations:
        if combination[0] != combination[1]:
            # True represents that the first point is better than the second in that criterium, otherwise it is False
            combs = [True]*len(card_winrates.columns)
            for k in range(len(combs)):
                if card_winrates.loc[combination[0], card_winrates.columns[k]] > card_winrates.loc[combination[1], card_winrates.columns[k]]:
                    combs[k] = False

            if all(combs):
                card_doms[combination[0]] += 1
            elif all(not x for x in combs):
                card_doms[combination[1]] += 1


            # comb1 = True
            # comb2 = True
            # if card_winrates.at[combination[0], "game_count"] >= card_winrates.at[combination[1], "game_count"] and \
            #         card_winrates.at[combination[0], "winrate"] >= card_winrates.at[combination[1], "winrate"]:
            #     comb2 = False
            # elif card_winrates.at[combination[0], "game_count"] <= card_winrates.at[combination[1], "game_count"] and \
            #             card_winrates.at[combination[0], "winrate"] <= card_winrates.at[combination[1], "winrate"]:
            #     comb1 = False
            #
            # if comb1:
            #     card_doms[combination[1]] += 1
            # elif comb2:
            #     card_doms[combination[0]] += 1

    for k in card_names:
        while len(fronts) < card_doms[k] + 1:
            fronts.append([])
        fronts[card_doms[k]].append(k)
    k = 0
    while k < len(fronts):
        if len(fronts[k]) == 0:
            fronts.remove([])
        else:
            k += 1

    return fronts


def create_graph(card_winrates, front, front_num, label = False):
    font = {'size': 5}
    matplotlib.rc('font', **font)

    points_1 = card_winrates.loc[front, 'game_count']
    points_2 = card_winrates.loc[front, 'winrate']

    plt.scatter(points_1, points_2)

    plt.xlabel('game_count')
    plt.ylabel('winrate')

    if label:
        if len(front)>1:
            for index, k in enumerate(front):
                plt.text(points_1.iloc[index], points_2.iloc[index], k + " " + str(front_num), va='bottom', ha='center')
        else:
            plt.text(points_1, points_2, front[0] + " " + str(front_num), va='bottom', ha='center')


if __name__ == "__main__":
    decks = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Decks',
                          usecols=['date', 'player', 'card'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'card': 'str'})
    decks.date = decks.date.apply(lambda x: x.date())

    games = pd.read_excel(io='./alahamaretov_arhiv/shoebox.xlsx',
                          sheet_name='Games',
                          usecols=['date', 'player', 'opponent', 'wins', 'losses'],
                          dtype={'date': 'datetime64[ns]', 'player': 'str', 'opponent': 'str', 'wins': 'int',
                                 'losses': 'int'})
    games.date = games.date.apply(lambda x: x.date())

    cube = pd.read_csv('./alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})

    alpha = 0
    # for each card in the current cube that was played at least once, lists its total game count and winrate
    query = ('SELECT cube.card, sum(wins) as wins, sum(losses) as losses, sum(wins)+sum(losses) as game_count, '
             f'(sum(wins)+{str(alpha)})/CAST((sum(wins)+sum(losses)+2*{str(alpha)}) as float) as winrate '
                 'FROM decks INNER JOIN games ON decks.date==games.date AND decks.player==games.player '
                 'INNER JOIN cube ON decks.card==cube.card '
                 'GROUP BY decks.card '
                 'ORDER BY game_count DESC')

    card_winrates = ps.sqldf(query)

    card_names = card_winrates['card'].tolist()
    card_winrates.set_index('card', inplace=True)

    for k in cube['card']:
        if k not in card_names:
            card_winrates.loc[k] = [0, 0, 0, 0]

    print(card_winrates)

    # create_graph(card_winrates)

    fronts = divide_into_fronts(card_winrates)

    for index, k in enumerate(fronts[:10]):
        create_graph(card_winrates, k, index, True)
    # for index, k in enumerate(fronts[-1:]):
    #     create_graph(card_winrates, k, len(fronts)-index, True)

    plt.savefig("pareto_fronts.png")
    plt.show()