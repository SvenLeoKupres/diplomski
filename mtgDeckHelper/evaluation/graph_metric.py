import pandas as pd
from matplotlib import pyplot as plt

if __name__=='__main__':
    # k = 360
    n_best = 5

    height = 18
    width = 18

    fig1, axs1 = plt.subplots(5, 5)
    # fig1.suptitle("Rezultati modela ovisno o dimenziji metričkog prostora")
    fig1.set_figheight(height)
    fig1.set_figwidth(width)
    fig2, axs2 = plt.subplots(5, 5)
    # fig1.suptitle("Rezultati modela ovisno o dimenziji metričkog prostora")
    fig2.set_figheight(height)
    fig2.set_figwidth(width)
    fig3, axs3 = plt.subplots(5, 5)
    # fig1.suptitle("Rezultati modela ovisno o dimenziji metričkog prostora")
    fig3.set_figheight(height)
    fig3.set_figwidth(width)

    for k in range(360, 601, 10):
        index1 = int(((k-360)/10) // 5)
        index2 = int(((k-360)/10) % 5)

        df = []
        for i in range(2, 6):
            df.append(pd.read_excel('./results.xlsx', sheet_name=f"{i}_{k}").iloc[:, 1:])

        df = sum(df) / 4
        df = df.transpose()

        arr1 = [df.loc[i, :].sum() for i in df.index.tolist()]
        axs1[index1, index2].plot(df.index, arr1, marker='o')
        axs1[index1, index2].set_title(f"br. karata={k}")
        # for ax in axs1.flat:
        #     ax.set(xlabel="Dimenzija metričkog prostora", ylabel="Broj pogođenih karata")

        df_sum = df.cumsum(axis=1)
        column_name = df_sum.columns[-1]
        top_n_rows = df_sum.nlargest(n_best, columns=column_name)

        arr = [i for i in range(1, 46)]
        for i in range(0, len(top_n_rows.index)):
            axs2[index1, index2].plot(arr, df_sum.iloc[i, :].tolist())  # , marker='o')
        axs2[index1, index2].set_title(f"br. karata={k}")
        axs2[index1, index2].legend([f"embedding_dim={i}" for i in top_n_rows.index.tolist()])

        df_2 = df * [15 - i % 15 for i in range(45)]
        df_2_sum = df_2.cumsum(axis=1)
        column_name = df_2_sum.columns[-1]
        top_n_rows = df_2_sum.nlargest(n_best, columns=column_name)
        print(f"num_cards={k}, best performer: embedding_dim={top_n_rows.index[-1]}, score={top_n_rows.iloc[-1, -1]}")
        for i in range(0, len(top_n_rows.index)):
            axs3[index1, index2].plot(arr, df_2_sum.iloc[i, :].tolist())  # , marker='o')
        axs3[index1, index2].set_title(f"br. karata={k}")
        axs3[index1, index2].legend([f"embedding_dim={i}" for i in top_n_rows.index.tolist()])

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(f"./graphs/metric_graph1.png")
    fig2.tight_layout()
    fig2.show()
    fig2.savefig(f"./graphs/metric_graph2.png")
    fig3.tight_layout()
    fig3.show()
    fig3.savefig(f"./graphs/metric_graph3.png")