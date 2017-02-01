import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from windrose import WindroseAxes
import numpy as np
# import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as sm

plt.style.use('classic')

save_fig = True


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
        df[period[0]:period[1]].plot(y=column, style='ks-', legend=False)
    else:
        df[period[0]:period[1]].plot(y=column, style='ko', legend=False)
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
    # plt.legend.remove()
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
    plot_graphs(df, r'PO4', r'PO4', 'mg/m3', lines=True)
    df.rename(columns={r'Inflow concentration as NH4 [mg m-3]': 'NH4'}, inplace=True)
    plot_graphs(df, r'NH4', r'NH4', 'mg/m3', lines=True)
    df.rename(columns={r'*Suspended sediment concentration [mg m-3]': 'Suspended'}, inplace=True)
    plot_graphs(df, r'Suspended', r'Suspended sediment concentration', 'mg/m3', lines=True)
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4193490_temp.csv', encoding="ISO-8859-1")
    df.rename(columns={r'Inflow temperature [°C]': 'inflowT'}, inplace=True)
    df.rename(columns={r'Inflow concentration of O2 [mg m-3]': 'inflowO2'}, inplace=True)
    df.rename(columns={r'Inflow pH [-]': 'inflowpH'}, inplace=True)
    plot_graphs(df, r'inflowT', r'Inflow Temperature', 'C')
    plot_graphs(df, r'inflowO2', r'Inflow O2', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowpH', r'Inflow pH', '-', lines=True)
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4193500_discharge.csv', encoding="ISO-8859-1")
    df.rename(columns={r'Inflow volume [m3 d-1]': 'inflowV'}, inplace=True)
    df.rename(columns={r'Suspended sediment concentration [mg m-3]': 'inflowSusp'}, inplace=True)
    plot_graphs(df, r'inflowV', r'Inflow Volume', 'm3/d')
    plot_graphs(df, r'inflowSusp', r'Inflow Suspended sediment', 'mg/m3', lines=True)
    df = load_data(r'../measurements/Excel Files/task 3/Maumee River/4193500.csv', encoding="ISO-8859-1")
    df.rename(columns={r'Inflow temperature [°C]': 'inflowT'}, inplace=True)
    plot_graphs(df, r'inflowT', r'Inflow Temperature', 'C')
    df.rename(columns={r'Inflow pH [-]': 'inflowpH'}, inplace=True)
    plot_graphs(df, r'inflowpH', r'Inflow pH', '-', lines=True)
    df.rename(columns={r'Suspended sediment concentration [mg m-3]': 'inflowSusp'}, inplace=True)
    df.rename(columns={r'Inflow concentration of O2 [mg m-3]': 'inflowO2'}, inplace=True)
    df.rename(columns={r'*Suspended solids, water, unfiltered [mg m-3]': 'inflowSuspUnfilt'}, inplace=True)
    plot_graphs(df, r'inflowO2', r'Inflow O2', 'mg/m3')
    # plot_graphs(df, r'inflowSuspUnfilt', r'Inflow Suspended solids, unfiltered', 'mg/m3')
    df.rename(columns={r'*Phosphorus, water, unfiltered, as phosphorus [mg m-3]': 'inflowP'}, inplace=True)
    plot_graphs(df, r'inflowP', r'Inflow P, unfiltered', 'mg/m3',lines=True)
    df.rename(columns={r'* Phosphorus, water, filtered, as phosphorus [mg m-3]': 'inflowPfil'}, inplace=True)
    df.rename(columns={r'* Orthophosphate, water, filtered,  as phosphorus [mg m-3]': 'inflowPO4filA'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved organic carbon (DOC) [mg m-3]': 'inflowDOC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved inorganic carbon (DIC) [mg m-3]': 'inflowDIC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of Ca2+ [mg m-3]': 'inflowCa'}, inplace=True)
    df.rename(columns={r'Inflow concentration of SO4 [mg m-3]': 'inflowSO4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved silica [mg m-3]': 'inflowDSi'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved silica [mg m-3]': 'inflowDSi'}, inplace=True)
    df.rename(columns={r'*Orthophosphate, water, filtered, as PO4 [mg m-3]': 'inflowPO4fil'}, inplace=True)
    df.rename(columns={r'*Iron, suspended sediment, recoverable [mg m-3]': 'inflowFeSuspSed'}, inplace=True)
    df.rename(columns={r'*Iron, water, unfiltered, recoverable [mg m-3]': 'inflowFe'}, inplace=True)
    df.rename(columns={r'*Iron, water, filtered [mg m-3]': 'inflowFefil'}, inplace=True)
    df.rename(columns={r'Inflow concentration of aluminum (Al3+) [mg m-3]': 'inflowAl'}, inplace=True)
    df.rename(columns={r'Inflow concentration of chlorophyll-a (Chla-P) [mg m-3]': 'inflowChlP'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NH4 [mg m-3]': 'inflowNH4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NO3 [mg m-3]': 'inflowNO3'}, inplace=True)
    plot_graphs(df, r'inflowPO4filA', r'Inflow P, filtered', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowPfil', r'Inflow P, filtered', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowDOC', r'Inflow DOC', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowDIC', r'Inflow DIC', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowCa', r'Inflow Ca2+', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowSO4', r'Inflow SO4', 'mg/m3',lines=True)
    df.inflowDSi[df.inflowDSi>450000]=0
    plot_graphs(df, r'inflowDSi', r'Inflow dSi', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowPO4fil', r'Inflow PO4', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowFeSuspSed', r'Inflow Fe suspended sediments', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowFe', r'Inflow Fe', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowFefil', r'Inflow Fe filtered', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowAl', r'Inflow Al3+', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowChlP', r'Inflow chlorophyll-a (Chla-P)', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowNH4', r'Inflow NH4', 'mg/m3',lines=True)
    plot_graphs(df, r'inflowNO3', r'Inflow NO3', 'mg/m3',lines=True)

if __name__ == '__main__':
    plotting_weather_wester_basin()

