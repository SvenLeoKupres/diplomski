import pandas as pd
from matplotlib import pyplot as plt
import re

def import_sheets(path, num_sheets):
    arr = []
    for k in range(num_sheets):
        df = pd.read_excel(path, sheet_name=f'Sheet{k+1}').iloc[:-1].transpose()
        df.set_index(df.iloc[:, 0])
        arr.append(df)

    return arr


def import_basic():
    return import_sheets('./by_hand/basic_color_bonus.ods', 4)


def import_pareto_basic():
    return import_sheets('./by_hand/pareto_basic_color_bonus.ods', 4)


def import_FE():
    return import_sheets('./by_hand/FE_color_bonus.ods', 4)


def import_FE_pareto():
    return import_sheets('./by_hand/pareto_FE_color_bonus.ods', 4)


def represent_by_hand():
    # df = import_basic()
    # df = import_pareto_basic()
    # df = import_FE()
    df = import_FE_pareto()

    df = sum(df)
    df /= 4
    df_sum = df.cumsum(axis=1)

    color_nums = []
    for k in df.index:
        color_nums.append(re.findall(r'\d+', k)[0])

    plt.title("Rezultati modela ovisno o jačini bonusa za boju karte")
    plt.ylabel("Broj pogođenih karata")
    plt.xlabel("Bonus za boju karte")
    arr1 = [df.loc[k, :].sum() for k in df.index.tolist()]
    plt.plot(color_nums, arr1, marker='o')
    plt.show()

    arr = [k for k in range(1, 46)]

    for k in range(len(df.index)):
        # print(df_1.iloc[:, k].tolist())
        # plt.plot(arr, df_1.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_2.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_3.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_4.iloc[k, :].tolist(), marker='o')
        plt.plot(arr, df_sum.iloc[k, :].tolist())  # , marker='o')
    plt.title("Ukupan broj točno pogođenih karata nakon svake odluke")
    plt.legend(df.index.tolist())
    plt.show()

    df_2 = df * [15 - k % 15 for k in range(45)]
    df_2_sum = df_2.cumsum(axis=1)
    for k in range(len(df.index)):
        # print(df_1.iloc[:, k].tolist())
        # plt.plot(arr, df_1.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_2.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_3.iloc[k, :].tolist(), marker='o')
        # plt.plot(arr, df_4.iloc[k, :].tolist(), marker='o')
        plt.plot(arr, df_2_sum.iloc[k, :].tolist())  # , marker='o')
    plt.title("Ukupan broj bodova nakon svake odluke")
    plt.legend(df.index.tolist())
    plt.show()


def only_color_bonus():
    num_tries = 5

    df = []
    for k in range(1, 1 + num_tries):
        df.append(pd.read_excel('./only_color_bonus/basic_results.ods', sheet_name=f"Sheet{k}"))
        # df.append(pd.read_excel('./only_color_bonus/fe_results.ods', sheet_name=f"Sheet{k}"))
        # df.append(pd.read_excel('./only_color_bonus/pareto_results.ods', sheet_name=f"Sheet{k}"))
        # df.append(pd.read_excel('./only_color_bonus/pareto_fe_results.ods', sheet_name=f"Sheet{k}"))
        # df.append(pd.read_excel('./only_color_bonus/metric_results.ods', sheet_name=f"Sheet{k}"))

    df = sum(df) / num_tries
    df = df.transpose()
    # print(df.loc[:, 0])
    # print(df.index)
    arr1 = [df.loc[k, :].sum() for k in df.index.tolist()]
    plt.plot(df.index, arr1, marker='o')
    plt.title("Rezultati modela ovisno o jačini bonusa za boju karte")
    plt.ylabel("Broj pogođenih karata")
    plt.xlabel("Bonus za boju karte")
    plt.show()

    df_sum = df.cumsum(axis=1)

    arr = [k for k in range(1, 46)]
    for k in range(0, len(df.index), 5):
        plt.plot(arr, df_sum.iloc[k, :].tolist())  # , marker='o')
    plt.title("Ukupan broj točno pogođenih karata nakon svake odluke")
    plt.legend(range(0, len(df.index), 5))
    plt.show()

    df_2 = df * [15 - k % 15 for k in range(45)]
    df_2_sum = df_2.cumsum(axis=1)
    for k in range(0, len(df.index), 5):
        plt.plot(arr, df_2_sum.iloc[k, :].tolist())  # , marker='o')
    plt.title("Ukupan broj bodova nakon svake odluke")
    plt.legend([f"color_num={i}" for i in range(0, len(df.index), 5)])
    plt.show()


def grid():
    # model = "basic_results"
    # model = "fe_results"
    # model = "pareto_results"
    model = "pareto_fe_results"

    n_best = 5

    num_tries = 5

    height = 12
    width = 12

    fig1, axs1 = plt.subplots(3, 2)
    fig1.suptitle("Rezultati modela ovisno o jačini bonusa za boju karte")
    fig1.set_figheight(height)
    fig1.set_figwidth(width)
    fig2, axs2 = plt.subplots(3, 2)
    fig2.suptitle("Ukupan broj točno pogođenih karata nakon svake odluke")
    fig2.set_figheight(height)
    fig2.set_figwidth(width)
    fig3, axs3 = plt.subplots(3, 2)
    fig3.suptitle("Ukupan broj bodova nakon svake odluke")
    fig3.set_figheight(height)
    fig3.set_figwidth(width)

    df = []
    for k in range(0, 6):
        index1 = k//2
        index2 = k%2
        df.append([])
        for i in range(1, 1+num_tries):
            df[k].append(pd.read_excel(f'./{model}_grid{i}.ods', sheet_name=f"Sheet{k}"))
            # df[k].append(pd.read_excel(f'./metric_results.ods', sheet_name=f"Sheet{k}"))
        df[k] = sum(df[k])/num_tries
        df[k] = df[k].transpose()
        arr1 = [df[k].loc[i, :].sum() for i in df[k].index.tolist()]
        axs1[index1, index2].plot(df[k].index, arr1, marker='o')
        axs1[index1, index2].set_title(f"alpha={k}")
        for ax in axs1.flat:
            ax.set(xlabel="Bonus za boju karte", ylabel="Broj pogođenih karata")

        df_sum = df[k].cumsum(axis=1)

        column_name = df_sum.columns[-1]
        top_n_rows = df_sum.nlargest(n_best, columns=column_name)

        arr = [i for i in range(1, 46)]
        for i in range(0, len(top_n_rows.index)):
            axs2[index1, index2].plot(arr, df_sum.iloc[i, :].tolist())  # , marker='o')
        axs2[index1, index2].set_title(f"alpha={k}")
        axs2[index1, index2].legend([f"color_num={i}" for i in top_n_rows.index.tolist()])

        df_2 = df[k] * [15 - i % 15 for i in range(45)]
        df_2_sum = df_2.cumsum(axis=1)
        column_name = df_2_sum.columns[-1]
        top_n_rows = df_2_sum.nlargest(n_best, columns=column_name)
        print(f"alpha={k}, best performer: color_num={top_n_rows.index[-1]}, score={top_n_rows.iloc[-1, -1]}")
        for i in range(0, len(top_n_rows.index)):
            axs3[index1, index2].plot(arr, df_2_sum.iloc[i, :].tolist())  # , marker='o')
        axs3[index1, index2].set_title(f"alpha={k}")
        axs3[index1, index2].legend([f"color_num={i}" for i in top_n_rows.index.tolist()])

    fig1.tight_layout()
    fig1.show()
    fig1.savefig(f"./graphs/{model}_graph1.png")
    fig2.tight_layout()
    fig2.show()
    fig2.savefig(f"./graphs/{model}_graph2.png")
    fig3.tight_layout()
    fig3.show()
    fig3.savefig(f"./graphs/{model}_graph3.png")


if __name__=='__main__':
    grid()
    # only_color_bonus()