import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from windrose import WindroseAxes
import numpy as np
# import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as sm

plt.style.use('classic')

save_fig = False


def linear_fit(df, column):
    return sm.ols(formula=column + " ~ num", data=df).fit()


def graph_size(h, v):
    plt.rcParams['figure.figsize'] = h, v
    font = {'family': 'serif',
            'weight': 'bold',
            'size': 14, }
    plt.rc('font', **font)
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot_graphs(df, column, lgnd, units, lines=True):
    df["TMP"] = df.index.values                # index is a DateTimeIndex
    df = df[df.TMP.notnull()]                  # remove all NaT values
    df.drop(["TMP"], axis=1, inplace=True)
    df = df.sort_index()
    period = [df.index.values[0], df.index.values[-1]]
    plt.figure()
    graph_size(9, 6)
    df = df[period[0]:period[1]][np.isfinite(df[column])]
    if lines:
        df[period[0]:period[1]].plot(y=column, color='k')
    else:
        df[period[0]:period[1]].plot(y=column, style='ko')
    result = linear_fit(df, column)
    prstd, iv_l, iv_u = wls_prediction_std(result)
    x = df[period[0]:period[1]].index.to_pydatetime()
    # plt.legend([lgnd])
    plt.grid()
    plt.ylabel(lgnd + ', ' + units)
    plt.xlabel('Year')
    plt.tight_layout()
    plt.plot(x, result.fittedvalues.values, 'r')
    lin_fit_str = r'y = %.2f + ( %.2e ) * year' % (result.params[0], result.params[1] * 365)
    plt.annotate(lin_fit_str, xy=(0.05, 0.93), xycoords='axes fraction', color='r')
    # fig.subplots_adjust(top=0.85)
    # ax.text(2, 2, r'an equation: $E=mc^2$', fontsize=15)
    # ax.axis([0, 10, 0, 10])
    # plt.plot(x, iv_u.values, 'r--')
    # plt.plot(x, iv_l.values, 'r--')
    if save_fig:
        plt.savefig('plots/input/western/' + lgnd + '.png', dpi=150)
    plt.show()


def plot_windrose(df):
    ax = WindroseAxes.from_ax()
    ax.box(df['WDIR'], df['WSPD'], bins=np.arange(0, 16, 3))
    # ax.bar(df['WDIR'], df['WSPD'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    if save_fig:
        plt.savefig('plots/input/western/windrose.png', dpi=150)
    plt.show()


def load_data(pth, parse_dates=[[0, 1, 2]], skiprows=None, encoding = "utf8"):
    df = pd.read_csv(pth, parse_dates=parse_dates, skiprows=skiprows, encoding=encoding)
    df.rename(columns={df.columns[0]: 'YY_MM_DD'}, inplace=True)
    # df['YY_MM_DD'] = df[['YEAR', 'MONTH', 'DAY']].apply(lambda s : datetime.datetime(*s),axis = 1)
    df['YY_MM_DD'] = pd.to_datetime(df['YY_MM_DD'], errors='coerce')
    df.head()
    df = df.convert_objects(convert_numeric=True)
    df = df.set_index('YY_MM_DD')
    df['num'] = range(0, len(df))
    df = df.drop(df.index[[0]])
    df["TMP"] = df.index.values                # index is a DateTimeIndex
    df = df[df.TMP.notnull()]                  # remove all NaT values
    df.drop(["TMP"], axis=1, inplace=True)
    return df


def plotting_weather_wester_basin():
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/western_basin_average.csv')
    plot_graphs(df, 'ATMP', 'Temperature', 'C')
    plot_graphs(df, 'WTMP', 'Water Temperature', 'C')
    plot_graphs(df, 'WSPD', 'Wind Speed', 'm/s')
    plot_graphs(df, 'PRES', 'Pressure', 'hPa')
    plot_windrose(df)
    df = load_data(r'../measurements/Excel Files/task 2/cloud cover/basin averages/western_basin_average.csv')
    df.rename(columns={r'Cloud Cover (Fraction)': 'cloud'}, inplace=True)
    df = df[df > 0].dropna()
    plot_graphs(df, r'cloud', r'Cloud Cover (Fraction)', '-')
    df = load_data(r'../measurements/Excel Files/task 2/precipitation/basin averages/western_basin_precipitation_average.csv')
    plot_graphs(df, 'PRCP', 'Precipitation', 'mm/day')
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4192500_discharge.csv')
    df.rename(columns={r'Inflow volume [m3 d-1]': 'inflow'}, inplace=True)
    plot_graphs(df, r'inflow', r'Inflow volume', 'm3/d')
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4192500.csv', encoding="ISO-8859-1")
    df.rename(columns={r'*Orthophosphate, water, filtered, as PO4 [mg m-3]': 'PO4'}, inplace=True)
    plot_graphs(df, r'PO4', r'PO4', 'mg/m3', lines=False)
    df.rename(columns={r'Inflow concentration as NH4 [mg m-3]': 'NH4'}, inplace=True)
    plot_graphs(df, r'NH4', r'NH4', 'mg/m3', lines=False)
    df.rename(columns={r'*Suspended sediment concentration [mg m-3]': 'Suspended'}, inplace=True)
    plot_graphs(df, r'Suspended', r'Suspended sediment concentration', 'mg/m3', lines=False)


if __name__ == '__main__':
    # plotting_weather_wester_basin()
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4192500.csv', encoding="ISO-8859-1")
    df.rename(columns={r'*Suspended sediment concentration [mg m-3]': 'Suspended'}, inplace=True)
    plot_graphs(df, r'Suspended', r'Suspended sediment concentration', 'mg/m3', lines=False)
