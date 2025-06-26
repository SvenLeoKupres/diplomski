import pandas as pd

if __name__ == '__main__':
    sets = pd.read_csv('./expansions.csv')
    sets.set_index('card', inplace=True)
    original = sets.copy()

    best_sets = []
    while len(sets)>0:
        sums = sets.sum().rename('sums')  # can see which sets have the highest number of cards in them
        sum_cards = sets.transpose().sum().rename('sums')  # can see number of different sets each card has appeared in

        cardsies = sum_cards[sum_cards == 1]
        cardsies = sets[sets.index.isin(cardsies.index)]
        cardsies_sets = cardsies.idxmax(axis=1).rename('set')
        if len(cardsies_sets)==0:
            break
        sets.drop(cardsies.index, inplace=True)
        sets.drop(cardsies_sets, axis=1, inplace=True)
        best_sets += cardsies_sets.tolist()

    best_sets = set(best_sets)
    sums = original.sum().rename('sums')  # can see which sets have the highest number of cards in them
    sum_cards = original.transpose().sum().rename('sums')  # can
    pass
