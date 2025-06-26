import pandas as pd

if __name__=='__main__':
    data = pd.read_csv('./draft_data_public.NEO.PremierDraft.csv',
                usecols=['draft_id', 'event_match_wins', 'event_match_losses', 'pick'],
                dtype={'draft_id': 'str', 'event_match_wins': 'int', 'event_match_losses': 'int', 'pick': 'str'})

    winrate = data.loc[:, ['draft_id', 'event_match_wins', 'event_match_losses']].drop_duplicates()
    winrate.set_index('draft_id', inplace=True)

    alpha = 0
    winrate['winrate'] = (winrate['event_match_wins']+alpha) / (winrate['event_match_losses']+winrate['event_match_wins']+2*alpha)

    decks = data.loc[:, ['draft_id', 'pick']]
    decks.set_index('draft_id', inplace=True)
    decks = pd.pivot_table(decks, index=decks.index, columns='pick', fill_value=0, aggfunc='size')

    decks = decks[winrate['event_match_losses'] + winrate['event_match_wins'] > 0]
    winrate = winrate[winrate['event_match_losses'] + winrate['event_match_wins'] > 0]
    Y = winrate['winrate'].to_numpy()
    X = decks.iloc[:, 1:].to_numpy()

    pass