import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from windrose import WindroseAxes
import numpy as np
plt.style.use('classic')


def graph_size(h, v):
    plt.rcParams['figure.figsize'] = h, v
    font = {'family': 'serif',
            'weight': 'bold',
            'size': 14, }
    plt.rc('font', **font)
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot_graphs(df, column, lgnd, units, period=['2000', '2017']):
    graph_size(6, 4)
    df[period[0]:period[1]].plot(y=column, color='k')
    plt.legend([lgnd])
    plt.grid()
    plt.ylabel(lgnd + ', ' + units)
    plt.xlabel('Year')
    # xticks = pd.date_range(start=period[0], end=period[1], freq=365)
    # ax.set_xticklabels([x.strftime('%a\n%d\n%h\n%Y') for x in xticks])
    # ax.set_xticklabels([], minor=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/input/' + lgnd + '.png', dpi=150)


def plot_windrose(df):
    ax = WindroseAxes.from_ax()
    ax.box(df['WDIR'], df['WSPD'], bins=np.arange(0, 16, 3))
    # ax.bar(df['WDIR'], df['WSPD'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    # plt.savefig('plots/input/windrose.png', dpi=150s
    plt.show()


def load_data(pth='../measurements/weather/National_Buoy_Data_Center/basin_average/western_basin_average.csv'):
    df = pd.read_csv(pth, parse_dates=[[0, 1, 2]])
    df = df.drop(df.index[[0]])
    df['YY_MM_DD'] = pd.to_datetime(df['YY_MM_DD'], errors='coerce')
    df.head()
    df = df.convert_objects(convert_numeric=True)
    df = df.set_index('YY_MM_DD')
    print(df.head())
    return df


if __name__ == '__main__':
    if 'df' not in locals():
        df = load_data()
    # plot_graphs(df, 'ATMP', 'Temperature', 'C')
    plot_graphs(df, 'WTMP', 'Water Temperature', 'C')
    # plot_graphs(df, 'PRES', 'Pressure', 'hPa')
    # plot_windrose(df)
