import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from datetime import datetime
from windrose import WindroseAxes
import numpy as np
# import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as sm
import os

import seaborn as sns
from matplotlib.colors import ListedColormap
sns.set_style("whitegrid")

save_fig = False


def linear_fit(df, column):
    return sm.ols(formula=column + " ~ num", data=df).fit()


def graph_size(h, v):
    plt.rcParams['figure.figsize'] = h, v
    font = {'family': 'serif',
            'weight': 'bold',
            'size': 14, }
    plt.rc('font', **font)
    # plt.rc('text', usetex=False)
    # plt.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot_graphs(df, column, lgnd, units, lines=True, style='-'):
    df["TMP"] = df.index.values                # index is a DateTimeIndex
    df = df[df.TMP.notnull()]                  # remove all NaT values
    df.drop(["TMP"], axis=1, inplace=True)
    df = df.sort_index()
    period = [df.index.values[0], df.index.values[-1]]
    plt.figure()
    graph_size(9, 6)
    df = df[period[0]:period[1]][np.isfinite(df[column])]
    if lines:
        df[period[0]:period[1]].plot(y=column, color=sns.xkcd_rgb["black"], lw=3, legend=False)
    else:
        df[period[0]:period[1]].plot(y=column, style='ko', legend=False)
    result = linear_fit(df, column)
    prstd, iv_l, iv_u = wls_prediction_std(result)
    x = df[period[0]:period[1]].index.to_pydatetime()
    # plt.legend([lgnd])
    plt.grid()
    plt.ylabel(lgnd + ', ' + units)
    plt.xlabel('Year')
    ax = plt.gca()
    ax.grid(linestyle='-', linewidth=0.2)
    plt.tight_layout()
    plt.plot(x, result.fittedvalues.values, color=sns.xkcd_rgb["red"], lw=3)
    lin_fit_y0 = r'$y_0$ = %.2e' % (result.params[0])
    lin_fit_k = r'$k$ = %.2e' % (result.params[1] * 365)
    plt.annotate(lin_fit_y0, xy=(0.79, 0.94), xycoords='axes fraction', color='r')
    plt.annotate(lin_fit_k, xy=(0.80, 0.88), xycoords='axes fraction', color='r')
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


def load_data(pth, parse_dates=[[0, 1, 2]], skiprows=None, encoding="utf8"):
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


def plotting_weather_western_basin():
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
    df.inflowP[df.inflowP > 3000] = 0
    plot_graphs(df, r'inflowP', r'Inflow P, unfiltered', 'mg/m3', lines=True)
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
    plot_graphs(df, r'inflowPO4filA', r'Inflow P, filtered', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowPfil', r'Inflow P, filtered', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowDOC', r'Inflow DOC', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowDIC', r'Inflow DIC', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowCa', r'Inflow Ca2+', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowSO4', r'Inflow SO4', 'mg/m3', lines=True)
    df.inflowDSi[df.inflowDSi > 450000] = 0
    plot_graphs(df, r'inflowDSi', r'Inflow dSi', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowPO4fil', r'Inflow PO4', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowFeSuspSed', r'Inflow Fe suspended sediments', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowFe', r'Inflow Fe', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowFefil', r'Inflow Fe filtered', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowAl', r'Inflow Al3+', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowChlP', r'Inflow chlorophyll-a (Chla-P)', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowNH4', r'Inflow NH4', 'mg/m3', lines=True)
    plot_graphs(df, r'inflowNO3', r'Inflow NO3', 'mg/m3', lines=True)


def plotting_basin_temperatures():
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/western_basin_average.csv')
    plot_graphs(df, 'ATMP', 'Air Temperature Western basin', 'C', style='k-')
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/central_basin_average.csv')
    plot_graphs(df, 'ATMP', 'Air Temperature Central basin', 'C', style='k-')
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/eastern_basin_average.csv')
    df = df[df.ATMP < 35]
    plot_graphs(df, 'ATMP', 'Air Temperature Eastern basin', 'C', style='k-')


def plot_graphs_in_subplot(df, column, lgnd, units, style='.', color=sns.xkcd_rgb["black"], ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_ylabel(lgnd + ', ' + units)
    ax.grid(linestyle='-', linewidth=0.2)
    period = [df.index.values[0], df.index.values[-1]]
    ax.set_xlim(period)
    try:
        ax.plot(df.index.values, df[column].values, style, color=color, lw=3)
        df["TMP"] = df.index.values                # index is a DateTimeIndex
        df = df[df.TMP.notnull()]                  # remove all NaT values
        df.drop(["TMP"], axis=1, inplace=True)
        df = df.sort_index()
        df = df[period[0]:period[1]][np.isfinite(df[column])]
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))

        if df.size > 0:
            result = linear_fit(df, column)
            prstd, iv_l, iv_u = wls_prediction_std(result)
            x = df.index.to_pydatetime()
            ax.plot(x, result.fittedvalues.values, color=sns.xkcd_rgb["red"], lw=2)
            lin_fit_y0 = r'$y_0$ = %.2e' % (result.params[0])
            lin_fit_k = r'$k$ = %.2e' % (result.params[1] * 365)
            ax.annotate(lin_fit_y0, xy=(0.71, 0.91), xycoords='axes fraction', color='r', fontsize=10)
            ax.annotate(lin_fit_k, xy=(0.73, 0.8), xycoords='axes fraction', color='r', fontsize=10)
        else:
            raise
    except:
        err = '%s\nNo data!!' % (lgnd)
        ax.annotate(err, xy=(0.2, 0.5), xycoords='axes fraction', color='k', fontsize=10)



def plot_subplots_for_river_inputs(basin, river):
    # basin = 'Western'
    # river = 'maumeeriver'
    print('%s, %s' % (river.title(), basin))

    df = load_data(r'../measurements/Excel Files/task 3/' + basin + ' Basin/' + river + '_average.csv', encoding="ISO-8859-1")

    for clmn in df.columns:
        df.rename(columns={clmn:clmn.strip().replace("*","")}, inplace=True)

    df.rename(columns={r'Inflow volume [m3 d-1]': 'inflowQ'}, inplace=True)
    df.rename(columns={r'Inflow temperature [°C]': 'inflowT'}, inplace=True)
    df.rename(columns={r'Suspended sediment concentration [mg m-3]': 'Susp'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as phosphorus [mg m-3]': 'PO4a'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as PO4 [mg m-3]': 'PO4b'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus [mg m-3]': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, filtered, as phosphorus [mg m-3]': 'PO4c'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved organic carbon (DOC) [mg m-3]': 'DOC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved inorganic carbon (DIC) [mg m-3]': 'DIC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of chlorophyll-a (Chla-P) [mg m-3]': 'Chla'}, inplace=True)
    df.rename(columns={r'Inflow concentration of O2 [mg m-3]': 'O2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NO3 [mg m-3]': 'NO3'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NH4 [mg m-3]': 'NH4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of SO4 [mg m-3]': 'SO4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of CH4 [mg m-3]': 'CH4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of aqueous iron (Fe2+) [mg m-3]': 'Fe2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of Ca2+ [mg m-3]': 'Ca2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of total solid iron (Fe3+) [mg m-3]': 'Fe3'}, inplace=True)
    df.rename(columns={r'Inflow concentration of aluminum (Al3+) [mg m-3]': 'Al3'}, inplace=True)
    df.rename(columns={r'Inflow pH [-]': 'pH'}, inplace=True)
    df.rename(columns={r'Suspended solids, water, unfiltered [mg m-3]': 'SuspUnfil'}, inplace=True)
    df.rename(columns={r'Iron, suspended sediment, recoverable [mg m-3]': 'IronSuspSed'}, inplace=True)
    df.rename(columns={r'Iron, water, unfiltered, recoverable [mg m-3]': 'IronUnfilRec'}, inplace=True)
    df.rename(columns={r'Iron, water, filtered [mg m-3]': 'IronFilRec'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved silica [mg m-3]': 'dSi'}, inplace=True)

    plt.close()
    fig, axes = plt.subplots(5, 5, sharex='col', figsize=(20, 10), dpi=150)
    # fig.title(basin + ' basin, ' + river.title())
    plot_graphs_in_subplot(df, 'inflowQ', 'Inflow Q', '$[m^3$ $d^{-1}]$', ax=axes[0, 0])
    plot_graphs_in_subplot(df, 'inflowT', 'Inflow T', '$C$', ax=axes[0, 1])
    plot_graphs_in_subplot(df, 'Susp', 'Susp. sediment', '$[mg$ $m^{-3}]$', ax=axes[3, 0])
    plot_graphs_in_subplot(df, 'PO4a', '$PO_4$, filt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 0])
    plot_graphs_in_subplot(df, 'PO4b', '$PO_4$, filt., as $PO_4$', '$[mg$ $m^{-3}]$', ax=axes[1, 1])
    plot_graphs_in_subplot(df, 'PO4c', '$P$, filt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 2])
    plot_graphs_in_subplot(df, 'PO4d', '$P$, unfilt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 3])
    plot_graphs_in_subplot(df, 'DOC', 'DOC', '$[mg$ $m^{-3}]$', ax=axes[0, 2])
    plot_graphs_in_subplot(df, 'DIC', 'DIC', '$[mg$ $m^{-3}]$', ax=axes[0, 3])
    plot_graphs_in_subplot(df, 'Chla', 'Chl-a (Chla-P)', '$[mg$ $m^{-3}]$', ax=axes[0, 4])
    plot_graphs_in_subplot(df, 'O2', '$O_2$', '$[mg$ $m^{-3}]$', ax=axes[2, 0])
    plot_graphs_in_subplot(df, 'NO3', '$NO_3$', '$[mg$ $m^{-3}]$', ax=axes[2, 1])
    plot_graphs_in_subplot(df, 'NH4', '$NH_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 2])
    plot_graphs_in_subplot(df, 'SO4', '$SO_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 3])
    plot_graphs_in_subplot(df, 'CH4', '$CH_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 4])
    plot_graphs_in_subplot(df, 'Fe2', '$Fe^{2+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 0])
    plot_graphs_in_subplot(df, 'Fe3', '$Fe^{3+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 1])
    plot_graphs_in_subplot(df, 'Ca2', '$Ca^{2+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 2])
    plot_graphs_in_subplot(df, 'Al3', '$Al^{3+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 3])
    plot_graphs_in_subplot(df, 'pH', 'pH', '-', ax=axes[1, 4])
    plot_graphs_in_subplot(df, 'SuspUnfil', 'Susp. solids, unfilt.', '$[mg$ $m^{-3}]$', ax=axes[3, 1])
    plot_graphs_in_subplot(df, 'IronSuspSed', 'Iron, susp. sed., recov', '$[mg$ $m^{-3}]$', ax=axes[3, 2])
    plot_graphs_in_subplot(df, 'IronUnfilRec', 'Iron, unfilt., recov.', '$[mg$ $m^{-3}]$', ax=axes[3, 3])
    plot_graphs_in_subplot(df, 'IronFilRec', 'Iron, filt., recov.', '$[mg$ $m^{-3}]$', ax=axes[3, 4])
    plot_graphs_in_subplot(df, 'dSI', 'Dissolved silica', '$[mg$ $m^{-3}]$', ax=axes[4, 4])
    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0)
    fig.savefig('plots/rivers/' + basin + ' basin/' + river + '.png', dpi=300)
    plt.close()
    return df


def parsing_folders_with_names():
    for basin in ['Eastern', 'Central', 'Western']:
        files = os.listdir('../measurements/Excel Files/task 3/' + basin + ' Basin')
        files.remove('Icon\r')
        for river in files:
            # print(river[:-12])
            yield plot_subplots_for_river_inputs(basin, river[:-12])

if __name__ == '__main__':
    dfs = list(parsing_folders_with_names())
    # df = list(parsing_folders_with_names())
    # df = plot_subplots_for_river_inputs()
    # df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/western_basin_average.csv')
    # plot_graphs(df, 'ATMP', 'Air Temperature Western basin', 'C', style='k-')
    # plot_subplots_for_river_inputs()
    # df = load_data(r'../measurements/Excel Files/task 3/Western Basin/maumeeriver_average.csv', encoding="ISO-8859-1")
    # df.rename(columns={r'Inflow volume [m3 d-1]': 'inflowQ'}, inplace=True)
    # plot_graphs(df, 'inflowQ', 'Inflow volume', '$[m^3$ $d^{-1}]$')
