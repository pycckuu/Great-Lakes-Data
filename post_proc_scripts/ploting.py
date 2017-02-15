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
sns.set_context("poster")

# SAVE_FIG = True
# SHOW_FIG = False

SAVE_FIG = True
SHOW_FIG = True

DPI = 150


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
        plt.savefig('plots/input/western/windrose.png', dpi=150)
    if SHOW_FIG:
        plt.show()


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
    df = df.convert_objects(convert_numeric=True)
    df = df.set_index('YY_MM_DD')
    df['num'] = range(0, len(df))
    df = df.drop(df.index[[0]])
    df["TMP"] = df.index.values                # index is a DateTimeIndex
    df = df[df.TMP.notnull()]                  # remove all NaT values
    df.drop(["TMP"], axis=1, inplace=True)
    df = df.sort_index()
    return df


def plotting_basin_temperatures():
    """Plots graphs of temperatures in 3 basins    """
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/western_basin_average.csv')
    plot_graphs(df, 'ATMP', 'Air Temperature Western basin', 'C', style='k-')
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/central_basin_average.csv')
    plot_graphs(df, 'ATMP', 'Air Temperature Central basin', 'C', style='k-')
    df = load_data('../measurements/weather/National_Buoy_Data_Center/basin_average/eastern_basin_average.csv')
    df = df[df.ATMP < 35]
    plot_graphs(df, 'ATMP', 'Air Temperature Eastern basin', 'C', style='k-')


def plotting_weather_basin(basin):
    """plots weather graphs for specific basin

    Args:
        basin (string): name of the basin
    """
    df = load_matching_files_df('../measurements/Excel Files/task 1/Basins averages/', basin)
    df2 = load_matching_files_df('../measurements/Excel Files/task 2/cloud cover/basin averages/', basin)
    df3 = load_matching_files_df('../measurements/Excel Files/task 2/precipitation/basin averages/', basin)
    df4 = load_matching_files_df('../measurements/Excel Files/task 2/relative humidity, solar radiation + others/basin averages/', basin, skiprows=9)
    df2.rename(columns={r'Cloud Cover (Fraction)': 'cloud'}, inplace=True)
    df2 = df2[df2 > 0].dropna()
    df3['TTL_PRCP'] = df3['SNOW'] + df3['PRCP']

    plt.close()
    fig, axes = plt.subplots(2, 4, sharex='col', figsize=(20, 10), dpi=150)
    plot_all_years_graph_in_ax_of_subplot(df, 'ATMP', 'Temperature', '$[C]$', ax=axes[0, 0])
    plot_all_years_graph_in_ax_of_subplot(df, 'WSPD', 'Wind Speed', '$[m$ $s^{-1}]$', ax=axes[0, 1])
    plot_all_years_graph_in_ax_of_subplot(df, 'PRES', 'Pressure', '$[hPa]$', ax=axes[0, 2])
    plot_all_years_graph_in_ax_of_subplot(df, 'WTMP', 'Water Temperature', 'C', ax=axes[0, 3])
    plot_all_years_graph_in_ax_of_subplot(df2, r'cloud', r'Cloud Cover (Fraction)', '$[-]$', ax=axes[1, 0])
    plot_all_years_graph_in_ax_of_subplot(df3, 'TTL_PRCP', 'Precipitation (Snow and Rain)', '$[mm$ $day^{-1}]$', ax=axes[1, 1])
    plot_all_years_graph_in_ax_of_subplot(df4, 'SRAD', 'Solar Radiation', '$[MJ$ $m^{-2}$ $day^{-1}]$', ax=axes[1, 2])
    plot_all_years_graph_in_ax_of_subplot(df4, 'RH2M', 'Relative Humidity', '$[\%]$', ax=axes[1, 3])
    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0)
    if SAVE_FIG:
        fig.savefig('plots/weather/' + basin + ' basin weather.png', dpi=DPI)
    if SHOW_FIG:
        plt.show()
    # plt.close()


def add_j_day_and_sort_df(df):
    df['j_day'] = df.index.to_series().dt.strftime('%j')
    df['year'] = df.index.to_series().dt.strftime('%Y')
    df = df.sort_values(by='j_day', ascending=1)
    df = df.reset_index()
    df['j_day'] = df['j_day'].apply(pd.to_numeric)
    return df


def plotting_weather():
    """ plots weather for all 3 basins"""
    for basin in ['eastern', 'central', 'western']:
        plotting_weather_basin(basin)


def plot_all_years_graph_in_ax_of_subplot(df, column, lgnd, units, style='.', color=sns.xkcd_rgb["black"], ax=None, time_lim=None):
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

    ax.set_ylabel(lgnd + ', ' + units)
    ax.grid(linestyle='-', linewidth=0.2)
    if time_lim is None:
        time_lim = [np.datetime64('1980-01-01T00:00:00.000000000'), np.datetime64('2016-12-31T00:00:00.000000000')]

    ax.set_xlim(time_lim)

    try:
        ax.plot(df.index.values, df[column].values, style, color=color, lw=3)
        df = df[np.isfinite(df[column])]
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


def plot_1yr_graph_in_ax_of_subplot(df, column, lgnd, units, style='.', color=sns.xkcd_rgb["black"], ax=None):
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

    ax.set_ylabel(lgnd + ', ' + units)
    ax.grid(linestyle='-', linewidth=0.2)
    ax.set_xlim([datetime.date(2016, 12, 31), datetime.date(2017, 12, 31)])
    try:
        start_date = datetime.date(2016, 12, 31)
        x = df['j_day'].values
        dates = [start_date + datetime.timedelta(float(xval)) for xval in x]
        ax.plot(dates, df[column].values, style, color=color, lw=3)
        # sns.boxplot(dates, df[column].values, ax=ax)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        # ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=(1, 15)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    except:
        err = '%s\nNo data!!' % (lgnd)
        ax.annotate(err, xy=(0.2, 0.5), xycoords='axes fraction', color='k', fontsize=10)


def prepare_river_dataframe(df):
    for clmn in df.columns:
        df.rename(columns={clmn: clmn.replace("*", "").strip()}, inplace=True)

    df.rename(columns={r'Inflow volume [m3 d-1]': 'inflowQ'}, inplace=True)
    df.rename(columns={r'Inflow temperature [°C]': 'inflowT'}, inplace=True)
    df.rename(columns={r'Suspended sediment concentration [mg m-3]': 'Susp'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as phosphorus [mg m-3]': 'PO4a'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered, as PO4 [mg m-3]': 'PO4b'}, inplace=True)
    df.rename(columns={r'Orthophosphate, water, filtered as PO4 [mg m-3]': 'PO4b'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus [mg m-3]': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus': 'PO4d'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, filtered, as phosphorus [mg m-3]': 'PO4c'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered,  as phosphorus [mg m-3]': 'PO4c'}, inplace=True)
    df.rename(columns={r'Phosphorus, water, unfiltered, as phosphorus [mg m-3].1': 'PO4c'}, inplace=True)
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
    df.rename(columns={r'Suspended solids, water, unfiltered [mg m-3]': 'SuspUnfil'}, inplace=True)
    df.rename(columns={r'Suspended solids, water, unfilterd [mg m-3]': 'SuspUnfil'}, inplace=True)
    df.rename(columns={r'Iron, suspended sediment, recoverable [mg m-3]': 'IronSuspSed'}, inplace=True)
    df.rename(columns={r'Iron, suspended sediment, recovered [mg m-3]': 'IronSuspSed'}, inplace=True)
    df.rename(columns={r'Iron, water, unfiltered, recoverable [mg m-3]': 'IronUnfilRec'}, inplace=True)
    df.rename(columns={r'Iron, water, unfiltered, recovered [mg m-3]': 'IronUnfilRec'}, inplace=True)
    df.rename(columns={r'Iron, water, filtered [mg m-3]': 'IronFilRec'}, inplace=True)
    df.rename(columns={r'Inflow concentration of dissolved silica [mg m-3]': 'dSi'}, inplace=True)
    df = df.groupby(df.columns, axis=1).sum()
    return df


def fill_in_river_subplots(method_of_plot, df):
    plt.close()
    fig, axes = plt.subplots(5, 5, sharex='col', figsize=(20, 10), dpi=150)
    # fig.title(basin + ' basin, ' + river.title())
    method_of_plot(df, 'inflowQ', 'Inflow Q', '$[m^3$ $d^{-1}]$', ax=axes[0, 0])
    method_of_plot(df, 'inflowT', 'Inflow T', '$C$', ax=axes[0, 1])
    method_of_plot(df, 'Susp', 'Susp. sediment', '$[mg$ $m^{-3}]$', ax=axes[3, 0])
    method_of_plot(df, 'PO4a', '$PO_4$, filt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 0])
    method_of_plot(df, 'PO4b', '$PO_4$, filt., as $PO_4$', '$[mg$ $m^{-3}]$', ax=axes[1, 1])
    method_of_plot(df, 'PO4c', '$P$, filt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 2])
    method_of_plot(df, 'PO4d', '$P$, unfilt., as $P$', '$[mg$ $m^{-3}]$', ax=axes[1, 3])
    method_of_plot(df, 'DOC', 'DOC', '$[mg$ $m^{-3}]$', ax=axes[0, 2])
    method_of_plot(df, 'DIC', 'DIC', '$[mg$ $m^{-3}]$', ax=axes[0, 3])
    method_of_plot(df, 'Chla', 'Chl-a (Chla-P)', '$[mg$ $m^{-3}]$', ax=axes[0, 4])
    method_of_plot(df, 'O2', '$O_2$', '$[mg$ $m^{-3}]$', ax=axes[2, 0])
    method_of_plot(df, 'NO3', '$NO_3$', '$[mg$ $m^{-3}]$', ax=axes[2, 1])
    method_of_plot(df, 'NH4', '$NH_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 2])
    method_of_plot(df, 'SO4', '$SO_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 3])
    method_of_plot(df, 'CH4', '$CH_4$', '$[mg$ $m^{-3}]$', ax=axes[2, 4])
    method_of_plot(df, 'Fe2', '$Fe^{2+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 0])
    method_of_plot(df, 'Fe3', '$Fe^{3+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 1])
    method_of_plot(df, 'Ca2', '$Ca^{2+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 2])
    method_of_plot(df, 'Al3', '$Al^{3+}$', '$[mg$ $m^{-3}]$', ax=axes[4, 3])
    method_of_plot(df, 'pH', 'pH', '-', ax=axes[1, 4])
    method_of_plot(df, 'SuspUnfil', 'Susp. solids, unfilt.', '$[mg$ $m^{-3}]$', ax=axes[3, 1])
    method_of_plot(df, 'IronSuspSed', 'Iron, susp. sed., recov', '$[mg$ $m^{-3}]$', ax=axes[3, 2])
    method_of_plot(df, 'IronUnfilRec', 'Iron, unfilt., recov.', '$[mg$ $m^{-3}]$', ax=axes[3, 3])
    method_of_plot(df, 'IronFilRec', 'Iron, filt., recov.', '$[mg$ $m^{-3}]$', ax=axes[3, 4])
    method_of_plot(df, 'dSI', 'Dissolved silica', '$[mg$ $m^{-3}]$', ax=axes[4, 4])
    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=0)
    return fig


def plot_subplots_for_river_inputs(basin, river, method=plot_1yr_graph_in_ax_of_subplot):
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

    if method == plot_1yr_graph_in_ax_of_subplot:
        df = add_j_day_and_sort_df(df)

    plt.close()
    fig = fill_in_river_subplots(method, df)
    if SAVE_FIG:
        fig.savefig('plots/rivers/' + basin + ' basin/' + method.__name__[:8] + ' ' + river + '.png', dpi=DPI)

    if SHOW_FIG:
        plt.show()
    plt.close()
    return df


def plotting_river_input(method_to_plot=plot_1yr_graph_in_ax_of_subplot):
    """finds csv files in folders and plots graphs for rivers

    Args:
        method_to_plot (TYPE, optional): plot_1yr_graph_in_ax_of_subplot or plot_all_years_graph_in_ax_of_subplot
    """

    for basin in ['Eastern', 'Central', 'Western']:
        files = os.listdir('../measurements/Excel Files/task 3/' + basin + ' Basin')
        if 'Icon\r' in files:
            files.remove('Icon\r')
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for river in files:
            # print(river[:-12])
            plot_subplots_for_river_inputs(basin, river[:-12], method_to_plot)


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
        df_from_each_file = add_j_day_and_sort_df(df_from_each_file)
        df_from_each_file = df_from_each_file.reset_index()
        df_from_each_file['river'] = f[49:-12]
        dfs.append(df_from_each_file)
    dfs_new = pd.concat(dfs, ignore_index=True)
    return dfs_new


def plot_all_rivers_in_single_subplot():
    df = find_all_csv_in_subfolders_and_create_single_df("../measurements/Excel Files/task 3/")

    # fig = fill_in_river_subplots(plot_all_rivers_1yr_graph_in_ax_of_subplot, df)
    # if SAVE_FIG:
    #     fig.savefig('plots/rivers/all_rivers_1yr.png', dpi=DPI)

    # if SHOW_FIG:
    #     plt.show()
    # plt.close()
    return df


def plot_all_rivers_1yr_graph_in_ax_of_subplot(df, column, lgnd, units, style='.', color=sns.xkcd_rgb["black"], ax=None):
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

    # try:
    sns.factorplot(x="j_day", y=column, data=df, kind='box', size=6, aspect=2, palette=sns.color_palette("Blues"), linewidth=0.5, ax=ax)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
    ax.set_ylabel(lgnd + ', ' + units)
    ax.set_xticks(np.arange(0, 366, 31))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_xlabel('Month')
    ax.grid(linestyle='-', linewidth=0.2)

if __name__ == '__main__':
    # df = load_matching_files_df('../measurements/Excel Files/task 3/', '*')

    # df = plot_all_rivers_in_single_subplot()
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=150)
    # sns.factorplot(x="j_day", y="O2", data=df, kind='box', size=6, aspect=2, palette=sns.color_palette("Blues"), linewidth=0.5)
    plot_all_rivers_1yr_graph_in_ax_of_subplot(df, 'O2', 'Inflow concentration $O_2$', '$[mg$ $m^{-3}]$', ax=ax)
    # plt.show()
    plt.close()
    # plotting_river_input(plot_1yr_graph_in_ax_of_subplot)
    # plotting_river_input(plot_all_years_graph_in_ax_of_subplot)
    # plt.figure()
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
