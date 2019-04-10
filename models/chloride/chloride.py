# %%
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmocean
import os
import warnings
import pandas as pd
import h5py
from datetime import datetime
from scipy import interpolate
sys.path.append('/Users/imarkelo/git/PorousMediaLab/')
from porousmedialab.batch import Batch

sns.set_style("whitegrid")
sns.set_style("ticks")
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_style("ticks")
rc('text', usetex=False)
rc("savefig", dpi=90)
rc("figure", dpi=90)
plt.rcParams['figure.figsize'] = 6, 4
pd.options.display.max_columns = 999
pd.options.display.max_rows = 400
str(datetime.now())
%matplotlib osx



rivers_cl_kta = pd.read_csv('models/chloride/rivers_cl_kta.csv')

# %%
rivers = {'ca': {'clair': ['Thames_R', 'Sydenham_R', 'Ruscom_R'],
                 'eastern': ['Grand_R', 'Nanticoke_R', 'Lynn_R', 'Big_R'],
                 'central': ['Big_Otter_R', 'Kettle_R'],
                 'western': ['Turkey_R', 'Canard_R']},
          'us': {'clair': ['Clinton_R', 'Belle_R', 'Black_R_MI'],
                 'eastern': ['Cattaraugus_R', 'Buffalo_R'],
                 'central': ['Sandusky_R', 'Black_OH_R', 'Vermilion_R', 'Rocky_R', 'Cuyahoga_R',
                             'Chagrin_R', 'Grand_OH_R', 'Conneaut_R'],
                 'western': ['Rouge_R', 'Huron_MI_R', 'Raisin_R', 'Maumee_R', 'Portage_R']},
          'StClair': ['St_Clair_R']}
Lsc = rivers_cl_kta[
    ['Thames_R', 'Sydenham_R', 'Ruscom_R']+
    ['Clinton_R', 'Black_R_MI'] +
    ['St_Clair_R']
].dropna().sum(axis=1)

Lsc.plot()
plt.show()


# %%
discharge_ckmy = pd.read_csv('models/chloride/discharge_ckmy.csv')
Wsc = discharge_ckmy[rivers['us']['clair']+rivers['ca']
                              ['clair']+rivers['StClair']].dropna().sum(axis=1)
Wsc.plot()
plt.show()

# %%

shift = 7*12

tend = 22*12 - shift
dt = 0.1
bl = Batch(tend, dt)
bl.add_species(name='Csc', init_conc=8.35)
bl.constants['Vsc'] = 4.17

bl.functions['LTsc'] = np.array2string(Lsc.index.values-shift, separator=',')
bl.functions['Lsc'] = np.array2string(Lsc.values, separator=',')
bl.functions['WTsc'] = np.array2string(Wsc.index.values-shift, separator=',')
bl.functions['Wsc'] = np.array2string(Wsc.values, separator=',')
bl.functions['Wsc_spl'] = 'sp.interpolate.InterpolatedUnivariateSpline(WTsc, Wsc,ext=3)'
bl.functions['Lsc_spl'] = 'sp.interpolate.InterpolatedUnivariateSpline(LTsc, Lsc, ext=3)'


bl.dcdt['Csc'] = 'Lsc_spl(TIME)/Vsc  - Wsc_spl(TIME)*Csc/Vsc'
bl.solve()
bl.plot_profiles()


# %%

bl.plot_profiles()
plt.ylim(0,10)


# %%

WTsc = Wsc.index.values
sp.interpolate.InterpolatedUnivariateSpline(WTsc, Wsc, ext=3)
plt.plot(WTsc, Wsc)

# %%
LTsc = Lsc.index.values
Lsc_spl = sp.interpolate.InterpolatedUnivariateSpline(LTsc, Lsc, ext=3)
plt.scatter(LTsc, Lsc)
plt.plot(LTsc, Lsc_spl(LTsc))
# %%

print(bl.dynamic_functions['dydt_str'])
