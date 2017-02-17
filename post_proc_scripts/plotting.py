import matplotlib.pyplot as plt
# import matplotlib
import pandas as pd
# from datetime import datetime
from windrose import WindroseAxes
import numpy as np
# import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import statsmodels.formula.api as sm
import glob
import os

import matplotlib.ticker as tkr
import matplotlib.dates as mdates
import datetime


import seaborn as sns
# from matplotlib.colors import ListedColormap
sns.set_style("whitegrid")
sns.set_style("ticks")

# SAVE_FIG = True
# SHOW_FIG = False

SAVE_FIG = True
SHOW_FIG = True

DPI = 150

LEGEND = {
    'inflowQ': 'Inflow Q, [m3 d-1]',
    'inflowT': 'Inflow T, [°C]',
    'Susp': 'Susp. sed., $[mg$ $m^{-3}]$',
    'PO4a': '$PO_4$, filt., as $P$, $[mg$ $m^{-3}]$',
    'PO4b': '$PO_4$, filt., as $PO_4$, $[mg$ $m^{-3}]$',
    'PO4d': '$P$, unfl., as $P$, $[mg$ $m^{-3}]$',
    'PO4c': '$P$, filt., as $P$, $[mg$ $m^{-3}]$',
    'DOC': '$DOC$, $[mg$ $m^{-3}]$',
    'DIC': '$DIC$, $[mg$ $m^{-3}]$',
    'Chla': '$Chla-P$, $[mg$ $m^{-3}]$',
    'O2': '$O_2$, $[mg$ $m^{-3}]$',
    'NO3': '$NO_3$, $[mg$ $m^{-3}]$',
    'NH4': '$NH_4$, $[mg$ $m^{-3}]$',
    'SO4': '$SO_4$, $[mg$ $m^{-3}]$',
    'CH4': '$CH_4$, $[mg$ $m^{-3}]$',
    'Fe2': '$Fe^{2+}$, $[mg$ $m^{-3}]$',
    'Ca2': '$Ca^{2+}$, $[mg$ $m^{-3}]$',
    'Fe3': '$Fe^{3+}$, $[mg$ $m^{-3}]$',
    'Al3': '$Al^{3+}$, $[mg$ $m^{-3}]$',
    'pH': 'pH, [-]',
    'SuspUnfil': 'Susp. solids, unfl., $[mg$ $m^{-3}]$',
    'IronSuspSed': 'Iron, susp. sed., $[mg$ $m^{-3}]$',
    'IronUnfilRec': 'Iron, unfl., $[mg$ $m^{-3}]$',
    'IronFilRec': 'Iron, filt., $[mg$ $m^{-3}]$',
    'dSi': '$Si$, $[mg$ $m^{-3}]$',
    'ATMP': 'Temperature, $[C]$',
    'ATMP_1': 'Temperature, $[C]$',
    'T2M Average Air Temperature At 2 m Above The Surface Of The Earth (degrees C)': 'Temperature, $[C]$',
    'WSPD': 'Wind Speed, $[m$ $s^{-1}]$',
    'WSPD_1': 'Wind Speed at 10m, $[m$ $s^{-1}]$',
    'PRES': 'Pressure, $[hPa]$',
    'WTMP': 'Water Temperature, $[C]$',
    'cloud': r'Cloud Cover (Fraction), $[-]$',
    'TTL_PRCP': 'Precipitation (Snow and Rain), $[mm$ $day^{-1}]$',
    'SRAD': 'Solar Radiation, $[MJ$ $m^{-2}$ $day^{-1}]$',
    'RH2M': 'Relative Humidity, $[\%]$'}


def linear_fit(df, y, x='num'):
    """ finds parameters of linear fit

    Args:
        df (pd.dataframe): dataframe with data
        y (string): which column analyze?

    Returns:
        sm.ols: statistical linear model
    """
    return sm.ols(formula=y + " ~ " + x, data=df).fit()


def graph_size(h, v):
    """ adjusts figure size and parameters

    Args:
        h (float): Description
        v (float): Description
    """
    plt.rcParams['figure.figsize'] = h, v
    font = {'family': 'serif',
            'weight': 'bold',
            'size': 14, }
    plt.rc('font', **font)
    # plt.rc('text', usetex=False)
    # plt.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot_windrose(df):
    """Plots windrose """
    ax = WindroseAxes.from_ax()
    ax.box(df['WDIR'], df['WSPD'], bins=np.arange(0, 16, 3))
    # ax.bar(df['WDIR'], df['WSPD'], normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    if SAVE_FIG:
        plt.savefig('plots/input/western/windrose.png', dpi=DPI)
    if SHOW_FIG:
        plt.show()


def load_matching_files_df(path, wildcard, skiprows=None):
    """loads all matched files in 1 dataframe

    Args:
        path (TYPE): path to the folder with files
        wildcard (TYPE): name starts with this string
        skiprows (None, optional): how many rows to skip in the file

    Returns:
        dataframe: merged dataframe
    """
    all_files = glob.glob(os.path.join(path, wildcard + "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = (load_data(f, skiprows=skiprows) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    return concatenated_df


def find_all_csv_in_subfolders_and_create_single_df(pth, skiprows=None):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(pth):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            all_files.append(os.path.join(dirpath, filename))
    dfs = []
    for f in all_files:
        df_from_each_file = load_data(f, skiprows=skiprows, encoding="ISO-8859-1")
        df_from_each_file = prepare_river_dataframe(df_from_each_file)
        df_from_each_file['river'] = f[49:-12]
        dfs.append(df_from_each_file)

    dfs_new = pd.concat(dfs, ignore_index=True)

    return dfs_new


def load_data(pth, parse_dates=[[0, 1, 2]], skiprows=None, encoding="utf8"):
    """Loads the data from csv files

    Args:
        pth (string): path to file
        parse_dates (list, optional): columns of dates
        skiprows (None, optional): how many rows to skip
        encoding (str, optional): encoding

    Returns:
        TYPE: returns pandas dataframe
    """
    df = pd.read_csv(pth, parse_dates=parse_dates, skiprows=skiprows, encoding=encoding)
    df.rename(columns={df.columns[0]: 'YY_MM_DD'}, inplace=True)
    # df['YY_MM_DD'] = df[['YEAR', 'MONTH', 'DAY']].apply(lambda s : datetime.datetime(*s),axis = 1)
    df['YY_MM_DD'] = pd.to_datetime(df['YY_MM_DD'], errors='coerce')
    a = df['YY_MM_DD']
    df = df[df.YY_MM_DD.notnull()]
    df = df.convert_objects(convert_numeric=True)
    df['YY_MM_DD'] = a
    df = df.sort_values('YY_MM_DD')
    df['num'] = df['YY_MM_DD'] - datetime.datetime.strptime('1980-01-01', '%Y-%m-%d')
    df['num'] = (df['num'] / np.timedelta64(1, 'D')).astype(float)
    df['j_day'] = df['YY_MM_DD'].dt.strftime('%j')
    df['year'] = df['YY_MM_DD'].dt.strftime('%Y')
    df['j_day'] = df['j_day'].apply(pd.to_numeric)
    return df


def plot_all_years_graph_in_ax_of_subplot(df, column, style='.', color=sns.xkcd_rgb["black"], ax=None, time_lim=None):
    """this function return graph in subplot figure

    Args:
        df (pd.dataframe): pandas df
        column (string): specify which column to plot
        lgnd (string): legend on the plot
        units (string): units on y-axis
        style (str, optional): plt style, line or points
        color (string, optional): plt color
        ax (None, optional): ax of subplot
        time_lim (None, optional): time x-limits
    """
    if ax is None:
        ax = plt.gca()

    if column in LEGEND:
        ax.set_ylabel(LEGEND[column])
    else:
        ax.set_ylabel(column)

    ax.grid(linestyle='-', linewidth=0.2)
    if time_lim is None:
        time_lim = [np.datetime64('1980-01-01T00:00:00.000000000'), np.datetime64('2016-12-31T00:00:00.000000000')]

    ax.set_xlim(time_lim)

    try:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
        ax.plot(df['YY_MM_DD'].values, df[column].values, style, color=color, lw=3)
        df = df[pd.notnull(df[column])]
        result = linear_fit(df, column)
        ax.plot(df['YY_MM_DD'].values, result.fittedvalues.values, color=sns.xkcd_rgb["red"], lw=2)
        lin_fit_y0 = r'$y_0$ = %.2e' % (result.params[0])
        lin_fit_k = r'$k$ = %.2e' % (result.params[1] * 365)
        ax.annotate(lin_fit_y0, xy=(0.71, 0.91), xycoords='axes fraction', color='r', fontsize=10)
        ax.annotate(lin_fit_k, xy=(0.73, 0.8), xycoords='axes fraction', color='r', fontsize=10)
    except:
        err = '%s\nNo data!!' % (column)
        ax.annotate(err, xy=(0.2, 0.5), xycoords='axes fraction', color='k', fontsize=10)


def plot_1yr_graph_in_ax_of_subplot(df, column, style='.', color=sns.xkcd_rgb["black"], ax=None):
    """this function return graph in subplot figure overlying the data in Julian day

    Args:
        df (pd.dataframe): pandas df
        column (string): specify which column to plot
        lgnd (string): legend on the plot
        units (string): units on y-axis
        style (str, optional): plt style, line or points
        color (string, optional): plt color
        ax (None, optional): ax of subplot
    """
    if ax is None:
        ax = plt.gca()

    if column in LEGEND:
        ax.set_ylabel(LEGEND[column])
    else:
        ax.set_ylabel(column)

    ax.grid(linestyle='-', linewidth=0.2)
    ax.set_xlim([datetime.date(2016, 12, 31), datetime.date(2017, 12, 31)])
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    try:
        start_date = datetime.date(2016, 12, 31)
        x = df['j_day'].values
        dates = [start_date + datetime.timedelta(float(xval)) for xval in x]
        ax.plot(dates, df[column].values, style, color=color, lw=3)
    except:
        err = '%s\nNo data!!' % (column)
        ax.annotate(err, xy=(0.2, 0.5), xycoords='axes fraction', color='k', fontsize=10)


def plot_1yr_boxplot_in_ax_of_subplot(df, column, ax=None):
    """this function return graph in subplot figure overlying the data in Julian day

    Args:
        df (pd.dataframe): pandas df
        column (string): specify which column to plot
        lgnd (string): legend on the plot
        units (string): units on y-axis
        style (str, optional): plt style, line or points
        color (string, optional): plt color
        ax (None, optional): ax of subplot
    """
    if ax is None:
        ax = plt.gca()

    if column == 'river':
        df = df[(df['Susp'] > 0) | (df['PO4a'] > 0) | (df['PO4b'] > 0) | (df['PO4d'] > 0) | (df['PO4c'] > 0) | (df['DOC'] > 0) | (df['DIC'] > 0) | (df['Chla'] > 0) | (df['O2'] > 0) | (df['NO3'] > 0) | (df['NH4'] > 0) | (df['SO4'] > 0) | (df['Ca2'] > 0) | (df['Al3'] > 0) | (df['SuspUnfil'] > 0) | (df['IronSuspSed'] > 0) | (df['IronUnfilRec'] > 0) | (df['IronFilRec'] > 0) | (df['dSi'] > 0)]

    try:
        sns.boxplot(x="j_day", y=column, data=df, palette=sns.color_palette("Blues", 21), linewidth=0.5, ax=ax)

        if column in LEGEND:
            ax.set_ylabel(LEGEND[column])
        else:
            ax.set_ylabel(column)
        ax.set_xticks(np.arange(0, 366, 31))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_xlabel('')
        ax.grid(linestyle='-', linewidth=0.2)
    except:
        err = '%s\nNo data!!' % (column)
        ax.annotate(err, xy=(0.2, 0.5), xycoords='axes fraction', color='k', fontsize=10)


def prepare_river_dataframe(df):
    for clmn in df.columns:
        df.rename(columns={clmn: clmn.replace("*", "").strip()}, inplace=True)

    df.rename(columns={r'Inflow volume [m3 d-1]': 'inflowQ'}, inplace=True)
    df.rename(columns={r'Inflow volume': 'inflowQ'}, inplace=True)
    df.rename(columns={r'Inflow temperature [°C]': 'inflowT'}, inplace=True)
    df.rename(columns={r'Suspended sediment concentration [mg m-3]': 'Susp'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as phosphorus [mg m-3]': 'PO4a'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as PO4 [mg m-3]': 'PO4b'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered as PO4 [mg m-3]': 'PO4b'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus [mg m-3]': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered,  as phosphorus [mg m-3]': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus [mg m-3].1': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, filtered, as phosphorus [mg m-3]': 'PO4c'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved organic carbon (DOC) [mg m-3]': 'DOC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved inorganic carbon (DIC) [mg m-3]': 'DIC'}, inplace=True)
    df.rename(columns={r'Inflow concentration of chlorophyll-a (Chla-P) [mg m-3]': 'Chla'}, inplace=True)
    df.rename(columns={r'Inflow volume of chlorophyll-a (Chla-P) [mg m-3]': 'Chla'}, inplace=True)
    df.rename(columns={r'Inflow concentration of O2 [mg m-3]': 'O2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NO3 [mg m-3]': 'NO3'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NO3[mg m-3]': 'NO3'}, inplace=True)
    df.rename(columns={r'Inflow concentration of NH4 [mg m-3]': 'NH4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of SO4 [mg m-3]': 'SO4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of CH4 [mg m-3]': 'CH4'}, inplace=True)
    df.rename(columns={r'Inflow concentration of aqueous iron (Fe2+) [mg m-3]': 'Fe2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of Ca2+ [mg m-3]': 'Ca2'}, inplace=True)
    df.rename(columns={r'Inflow concentration of total solid iron (Fe3+) [mg m-3]': 'Fe3'}, inplace=True)
    df.rename(columns={r'Inflow concentration of aluminum (Al3+) [mg m-3]': 'Al3'}, inplace=True)
    df.rename(columns={r'Inflow pH [-]': 'pH'}, inplace=True)
    df.rename(columns={r'Inflow pH': 'pH'}, inplace=True)
    df.rename(columns={r'Suspended solids, water, unfiltered [mg m-3]': 'SuspUnfil'}, inplace=True)
    df.rename(columns={r'Suspended solids, water, unfilterd [mg m-3]': 'SuspUnfil'}, inplace=True)
    df.rename(columns={r'Iron, suspended sediment, recoverable [mg m-3]': 'IronSuspSed'}, inplace=True)
    df.rename(columns={r'Iron, suspended sediment, recovered [mg m-3]': 'IronSuspSed'}, inplace=True)
    df.rename(columns={r'Iron, water, unfiltered, recoverable [mg m-3]': 'IronUnfilRec'}, inplace=True)
    df.rename(columns={r'Iron, water, unfiltered, recovered [mg m-3]': 'IronUnfilRec'}, inplace=True)
    df.rename(columns={r'Iron, water, filtered [mg m-3]': 'IronFilRec'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved silica [mg m-3]': 'dSi'}, inplace=True)
    df = df.T.drop_duplicates()
    df = df.T
    for c in df.columns:
        if c is not 'YY_MM_DD':
            df[c] = df[c].apply(pd.to_numeric)
    return df


def fill_in_river_subplots(plotting_method, df):
    plt.close()
    fig, axes = plt.subplots(5, 5, sharex='col', figsize=(20, 10), dpi=DPI)
    plotting_method(df, 'inflowQ', ax=axes[0, 0])
    plotting_method(df, 'inflowT', ax=axes[0, 1])
    plotting_method(df, 'Susp', ax=axes[3, 0])
    plotting_method(df, 'PO4a', ax=axes[1, 0])
    plotting_method(df, 'PO4b', ax=axes[1, 1])
    plotting_method(df, 'PO4c', ax=axes[1, 2])
    plotting_method(df, 'PO4d', ax=axes[1, 3])
    plotting_method(df, 'DOC', ax=axes[0, 2])
    plotting_method(df, 'DIC', ax=axes[0, 3])
    plotting_method(df, 'Chla', ax=axes[0, 4])
    plotting_method(df, 'O2', ax=axes[2, 0])
    plotting_method(df, 'NO3', ax=axes[2, 1])
    plotting_method(df, 'NH4', ax=axes[2, 2])
    plotting_method(df, 'SO4', ax=axes[2, 3])
    plotting_method(df, 'CH4', ax=axes[2, 4])
    plotting_method(df, 'Fe2', ax=axes[4, 0])
    plotting_method(df, 'Fe3', ax=axes[4, 1])
    plotting_method(df, 'Ca2', ax=axes[4, 2])
    plotting_method(df, 'Al3', ax=axes[4, 3])
    plotting_method(df, 'pH', ax=axes[1, 4])
    plotting_method(df, 'SuspUnfil', ax=axes[3, 1])
    plotting_method(df, 'IronSuspSed', ax=axes[3, 2])
    plotting_method(df, 'IronUnfilRec', ax=axes[3, 3])
    plotting_method(df, 'IronFilRec', ax=axes[3, 4])
    plotting_method(df, 'dSI', ax=axes[4, 4])
    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0)
    return fig


def plotting_weather_basin(basin):
    """plots weather graphs for specific basin

    Args:
        basin (string): name of the basin
    """

    df = load_matching_files_df('../measurements/Excel Files/weather combinations/', basin)
    for clmn in df.columns:
        df.rename(columns={clmn: clmn.replace("*", "").strip()}, inplace=True)

    df.rename(columns={r'T2M Average Air Temperature At 2 m Above The Surface Of The Earth (degrees C)': 'ATMP_1'}, inplace=True)
    df.rename(columns={r'WIND Wind Speed At 10 m Above The Surface Of The Earth (m/s)': 'WSPD_1'}, inplace=True)
    df.rename(columns={r'WSPD (m/s)': 'WSPD'}, inplace=True)
    df.rename(columns={r'PRES (hPa)': 'PRES'}, inplace=True)
    df.rename(columns={r'WTMP (degC)': 'WTMP'}, inplace=True)
    df.rename(columns={r'Cloud Cover (Fraction)': 'cloud'}, inplace=True)
    df.rename(columns={r'Cloud Cover(fraction)': 'cloud'}, inplace=True)
    df.rename(columns={r'SRAD Daily Insolation Incident On A Horizontal Surface (MJ/m^2/day)': 'SRAD'}, inplace=True)
    df.rename(columns={r'RH2M Relative Humidity At 2 m (%)': 'RH2M'}, inplace=True)

    df['TTL_PRCP'] = df[['SNOW', 'PRCP', 'Total Rain (mm)', 'Total Snow (mm)']].sum(axis=1, skipna=True)
    df['WSPD_AV'] = df[['WSPD_1', 'WSPD']].mean(axis=1, skipna=True)

    for plotting_method in [plot_all_years_graph_in_ax_of_subplot, plot_1yr_graph_in_ax_of_subplot, plot_1yr_boxplot_in_ax_of_subplot]:
        plt.close()
        fig, axes = plt.subplots(2, 4, sharex='col', figsize=(20, 10), dpi=DPI)
        plotting_method(df, 'ATMP_1', ax=axes[0, 0])
        plotting_method(df, 'WSPD_1', ax=axes[0, 1])
        plotting_method(df[df.WSPD < 200], 'WSPD', ax=axes[0, 2])
        plotting_method(df, 'WTMP', ax=axes[0, 3])
        plotting_method(df[df.cloud > 0], 'cloud', ax=axes[1, 0])
        plotting_method(df, 'TTL_PRCP', ax=axes[1, 1])
        plotting_method(df, 'SRAD', ax=axes[1, 2])
        plotting_method(df, 'RH2M', ax=axes[1, 3])
        plt.tight_layout(pad=0.2, w_pad=1, h_pad=1)
        if SAVE_FIG:
            fig.savefig('plots/weather/' + plotting_method.__name__[5:14] + ' ' + basin + ' basin weather.png', dpi=DPI)
        if SHOW_FIG:
            plt.show()
        plt.close()


def plot_subplots_for_river_inputs(basin, river, plotting_method=plot_1yr_graph_in_ax_of_subplot):
    """Summary

    Args:
        basin (TYPE): Description
        river (TYPE): Description

    Returns:
        TYPE: Description
    """

    print('%s basin, %s' % (basin, river.title()))

    df = load_data(r'../measurements/Excel Files/task 3/' + basin + ' Basin/' + river + '_average.csv', encoding="ISO-8859-1")
    df = prepare_river_dataframe(df)

    plt.close()
    fig = fill_in_river_subplots(plotting_method, df)
    if SAVE_FIG:
        fig.savefig('plots/rivers/' + basin + ' basin/' + plotting_method.__name__[:8] + ' ' + river + '.png', dpi=DPI)

    if SHOW_FIG:
        plt.show()
    plt.close()
    return df


def plot_river_input():
    """finds csv files in folders and plots graphs for rivers

    Args:
        plotting_method (TYPE, optional): plot_1yr_graph_in_ax_of_subplot or plot_all_years_graph_in_ax_of_subplot
    """

    for basin in ['Eastern', 'Central', 'Western']:
        files = os.listdir('../measurements/Excel Files/task 3/' + basin + ' Basin')
        if 'Icon\r' in files:
            files.remove('Icon\r')
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for river in files:
            plot_subplots_for_river_inputs(basin, river[:-12], plot_1yr_graph_in_ax_of_subplot)
            plot_subplots_for_river_inputs(basin, river[:-12], plot_all_years_graph_in_ax_of_subplot)


def plot_weather():
    """ plots weather for all 3 basins"""
    for basin in ['eastern', 'central', 'western']:
        plotting_weather_basin(basin)


def plot_all_rivers_in_separate_plots(df=None):
    """ plot all rivers all years in separate graphs per column of dataframe

    Args:
        plotting_method (TYPE, optional): Description

    Returns:
        TYPE: Description
    """

    if df is None:
        df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/")
        # df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/Eastern basin")

    for plotting_method in [plot_all_years_graph_in_ax_of_subplot, plot_1yr_graph_in_ax_of_subplot, plot_1yr_boxplot_in_ax_of_subplot]:
        for c in df:
            if c not in ['YY_MM_DD', 'num', 'j_day', 'year']:
                f, a = plt.subplots(figsize=(10, 6), dpi=DPI)
                plotting_method(df, c, ax=a)

                if SAVE_FIG:
                    f.savefig('plots/rivers/all/' + plotting_method.__name__[5:14] + '/' + str(c) + '.png', dpi=DPI)

                if SHOW_FIG:
                    plt.show()
                plt.close()

    return df


def plot_all_rivers_in_single_subplot(df=None):
    """ plot all rivers all years in 1 subplots

    Args:
        plotting_method (TYPE, optional): Description

    Returns:
        TYPE: Description
    """

    if df is None:
        df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/")
        # df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/Eastern basin")

    for plotting_method in [plot_all_years_graph_in_ax_of_subplot, plot_1yr_graph_in_ax_of_subplot, plot_1yr_boxplot_in_ax_of_subplot]:
        plt.close()
        fig = fill_in_river_subplots(plotting_method, df)
        if SAVE_FIG:
            fig.savefig('plots/rivers/all/' + plotting_method.__name__[5:14] + ' ' + 'all_rivers_subplots.png', dpi=DPI)

        if SHOW_FIG:
            plt.show()
        plt.close()


def plot_density_of_measurements(df=None):

    if df is None:
        df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/Eastern basin")

    c = df.groupby(['j_day']).count()
    count = c.Susp + c.PO4a + c.PO4b + c.PO4d + c.PO4c + c.DOC + c.DIC + c.Chla + c.O2 + c.NO3 + c.NH4 + c.SO4 + c.Ca2 + c.Al3 + c.SuspUnfil + c.IronSuspSed + c.IronUnfilRec + c.IronFilRec + c.dSi

    f, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    ax.plot(count.index.values, count, '.', markersize=15)
    ax.set_xticks(np.arange(0, 366, 31))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlim((0, 364))
    ax.grid(linestyle='-', linewidth=0.2)
    ax.set_xlabel('Day of the year')
    ax.set_ylabel('Number of measurements')
    ax.annotate('Total amount: %s' % count.sum(), xy=(0.74, 0.95), xycoords='axes fraction', color='k', fontsize=15)

    if SAVE_FIG:
        f.savefig('plots/rivers/density_of_measurements.png', dpi=DPI)

    if SHOW_FIG:
        plt.show()
    plt.close()
    return df


def plot_completeness_of_measurements(df=None):

    if df is None:
        df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/")

    df = df[(df['Susp'] > 0) | (df['PO4a'] > 0) | (df['PO4b'] > 0) | (df['PO4d'] > 0) | (df['PO4c'] > 0) | (df['DOC'] > 0) | (df['DIC'] > 0) | (df['Chla'] > 0) | (df['O2'] > 0) | (df['NO3'] > 0) | (df['NH4'] > 0) | (df['SO4'] > 0) | (df['Ca2'] > 0) | (df['Al3'] > 0) | (df['SuspUnfil'] > 0) | (df['IronSuspSed'] > 0) | (df['IronUnfilRec'] > 0) | (df['IronFilRec'] > 0) | (df['dSi'] > 0)]

    f, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    sns.countplot(y='river', data=df, order=df.river.value_counts().iloc[:len(df.river.unique())].index, palette=sns.color_palette())
    ax.set_xlabel('Amount of measurements')
    ax.grid(linestyle='-', linewidth=0.2)
    ax.set_ylabel('River')
    # ax.annotate('Total amount: %s' % count.sum(), xy=(0.74, 0.95), xycoords='axes fraction', color='k', fontsize=15)

    if SAVE_FIG:
        f.savefig('plots/rivers/completeness_of_measurements.png', dpi=DPI)

    if SHOW_FIG:
        plt.show()
    plt.close()
    return df

if __name__ == '__main__':
    pass
    df = plot_completeness_of_measurements()
    plot_density_of_measurements(df)
    # plot_weather()
    # df = plot_all_rivers_in_separate_plots()
    # plot_density_of_measurements(df)
    # plot_river_input()
    # pass
    # df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/")
    # f, a = plt.subplots(figsize=(20, 10), dpi=DPI)
    # sns.boxplot(x="j_day", y='river', data=df, palette=sns.color_palette("Blues", df.river.unique().size), linewidth=0.5)
    # plt.show()
    # plt.close()
    # plt.close()
    # # df2 = df[df['river']=='blackriver'].pivot(index='j_day',columns='year', values='inflowT')
    # sns.tsplot(data=df, time='j_day', value='inflowQ', unit='year', n_boot=100)
    # plt.show()
    # # plt.savefig('test.png')
    # plt.close()
    # # Create a noisy periodic dataset
    # # sines = []
    # # rs = np.random.RandomState(8)
    # # for _ in range(15):
    # #     x = np.linspace(0, 30 / 2, 30)
    # #     y = np.sin(x) + rs.normal(0, 1.5) + rs.normal(0, .3, 30)
    # #     sines.append(y)

    # # # Plot the average over replicates with bootstrap resamples
    # # sns.tsplot(sines, err_style="boot_traces", n_boot=500)
    # # plt.show()
    # # plt.close()
    # # import numpy as np; np.random.seed(22)
    # # import seaborn as sns; sns.set(color_codes=True)
    # # x = np.linspace(0, 15, 31)
    # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
    # ax = sns.tsplot(data=data)
    # plt.show()
