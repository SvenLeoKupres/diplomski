import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import pandasql as ps

# from demos.fixed_effects_model import fe_model


def divide_into_fronts(card_winrates:pd.DataFrame):
    """
    :param card_winrates: pandas dataframe, indices are card names, all other columns are optimization criteria
    :return: pandas dataframe, indices are card names, values are indexes of the front the card is in
    """
    card_names = card_winrates.index

    criteria = card_winrates.columns

    combinations = pd.merge(card_winrates.reset_index(), card_winrates.reset_index(), how='cross')
    combinations = combinations[combinations['card_x']!=combinations['card_y']]
    tmp = [combinations[k+"_x"]>combinations[k+"_y"] for k in criteria]
    df = tmp[0].rename("truth").to_frame()
    for k in range(1, len(criteria)):
        df['truth'] *= tmp[k]

    tmp = df['truth'] == True
    combinations = combinations[tmp]['card_x'].to_frame()
    # combinations = combinations.groupby('card_x').apply(lambda x: x.count(), include_groups=False).rename(columns={"card_x": "count"})
    combinations['count'] = 0
    combinations = combinations.groupby('card_x').count()

    # cards that have yet to see any play have to be assigned a front as well
    tmp = combinations.index
    tmp = card_names[~card_names.isin(tmp)]
    df = pd.DataFrame({"card_x":tmp, "count":[0 for _ in tmp]})
    df.set_index('card_x', inplace=True)

    combinations = pd.concat([combinations, df])
    combinations = combinations.reset_index().rename(columns={"card_x": "card"}).set_index('card')

    k = 0
    while k<max(combinations['count']):
        num = combinations[combinations['count']==k+1]
        if len(num) == 0:
            combinations.loc[combinations['count']>k, 'count'] -= 1
        else:
            k += 1
    combinations['count'] = max(combinations['count']) - combinations['count']

    # fronts = []
    # card_winrates['doms'] = 0
    #
    # combinations = list(itertools.product(card_names, card_names))
    # for combination in combinations:
    #     if combination[0] != combination[1]:
    #         # True represents that the first point is better than the second in that criterium, otherwise it is False
    #         combs = [True]*len(card_winrates.columns)
    #         for k in range(len(combs)):
    #             if card_winrates.loc[combination[0], card_winrates.columns[k]] > card_winrates.loc[combination[1], card_winrates.columns[k]]:
    #                 combs[k] = False
    #
    #         if all(combs):
    #             card_winrates.at[combination[0], 'doms'] += 1
    #             # card_doms[combination[0]] += 1
    #         elif all(not x for x in combs):
    #             card_winrates.at[combination[1], 'doms'] += 1
    #             # card_doms[combination[1]] += 1
    #
    # for k in card_names:
    #     while len(fronts) < card_winrates.at[k, 'doms'] + 1:
    #         fronts.append([])
    #     fronts[card_winrates.at[k, 'doms']].append(k)
    # k = 0
    # while k < len(fronts):
    #     if len(fronts[k]) == 0:
    #         fronts.remove([])
    #     else:
    #         k += 1

    return combinations.rename(columns={'count': 'front'})


def create_graph(card_winrates:pd.DataFrame, fronts:pd.DataFrame, front_num, label:bool = False):
    """
    Creates a matplotlib graph showcasing the pareto fronts (only for 2 criteria)
    :param card_winrates: indices are card names, optimization criteria of the cards
    :param fronts: indices are fronts of the cards (first front is 0)
    :param front_num: index of the front
    :param label: true if you want to present labels for card names, false otherwise
    :return:
    """
    font = {'size': 5}
    matplotlib.rc('font', **font)

    df = pd.merge(fronts, card_winrates, on='card')

    cards = df[df['front']==front_num]

    column_x = card_winrates.columns[0]
    column_y = card_winrates.columns[1]

    points_1 = cards[column_x]
    points_2 = cards[column_y]

    plt.scatter(points_1, points_2)

    plt.xlabel(column_x)
    plt.ylabel(column_y)

    if label:
        for index, k in enumerate(cards.index):
            plt.text(points_1.iloc[index], points_2.iloc[index], k + " " + str(front_num), va='bottom', ha='center')


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

    # df = decks.merge(games, on=["date", "player"])
    # model = fe_model(df, 0)
    # diff = model.coef_.sum(axis=0)

    card_names = card_winrates['card'].tolist()
    card_winrates.set_index('card', inplace=True)

    for k in cube['card']:
        if k not in card_names:
            card_winrates.loc[k] = [0, 0, 0, 0]

    # print(card_winrates)

    # create_graph(card_winrates)

    card_winrates = card_winrates[['winrate', 'game_count']]

    fronts = divide_into_fronts(card_winrates)

    for k in range(20):
        create_graph(card_winrates, fronts, k, True)
    # for k in range(len(fronts)):
    #     create_graph(card_winrates, fronts, k, True)

    # plt.savefig("pareto_fronts.png")
    plt.show()