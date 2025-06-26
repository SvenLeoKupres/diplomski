import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
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

    cube = pd.read_csv('../alahamaretov_arhiv/cube.csv',
                       usecols=['name', 'CMC', 'Type', 'Color'],
                       dtype={'name': 'str', 'CMC': 'int', 'Type': 'str', 'Color': 'str'})
    cube = cube.rename(columns={"name": "card", "CMC": "cmc", "Type": "type", "Color": "color"})
    cube.set_index('card', inplace=True)

    reshaped = pd.pivot_table(
        decks,
        index=decks.index,  # since the index already includes 'date' and 'player'
        columns='card',
        aggfunc=lambda x: 1,  # set to 1 where card was used
        fill_value=0
    )

    # Clean up column names
    reshaped.columns.name = None
    reshaped.columns = [f'{col}' for col in reshaped.columns]

    # First, make sure index is a MultiIndex, not a single column of tuples
    reshaped.index = pd.MultiIndex.from_tuples(reshaped.index, names=['date', 'player'])

    # Then reset it to get 'date' and 'player' as separate columns
    reshaped = reshaped.reset_index()

    cardnames = decks.loc[:, "card"].sort_values().unique()
    not_played = cube[~cube.index.isin(cardnames)].index

    reshaped[not_played] = 0
    # print(reshaped)

    df = reshaped.iloc[:, 2:]
    # print(df)

    X = df.to_numpy()
    # print(X)

    alpha = 0
    df2 = games.groupby(['date', 'player']) # ['wins'].sum() #.apply(lambda x: x.wins/(x.wins+x.losses))
    deck_wins = df2['wins'].sum()
    deck_losses = df2['losses'].sum()
    deck_winrates = (deck_wins+alpha) / (deck_wins+deck_losses+2*alpha)

    y = deck_winrates.to_numpy()

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    print(((y_pred-y)**2).sum()/y.shape[0])
    pass
    # svaki koeficijent pripada jednoj karti, moze se koristiti kao ocjena za kartu - isto kao i u fixed effects modelu
    # bias nam ne sluzi nicemu


