import autoencodeSVJ.utils as utils
import autoencodeSVJ.trainer as trainer
import autoencodeSVJ.evaluate as ev

import pandas as pd
import datetime

print("imports OK")

ev.update_all_signal_evals()

aucs = ev.load_auc_table()
s = utils.summary()


consider = s.sort_values('start_time')[::-1].iloc[1:15].filename
a = aucs.loc[:,consider.values]

x = s[s.filename == 'hlf_eflow3_8_v36'].T

print(x)

s = utils.summary(include_outdated=True)

aucs.mean().sort_values()[::-1]

res = utils.summary()
res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime(year=2020, month=6, day=1, hour=22, minute=30)]
res = res[res.epochs == 100]
import matplotlib.pyplot as plt

plt.scatter(res.total_loss, res.mae_auc)
# plt.scatter(res.total_loss, res.mse_auc)
#plt.show()


#reload(utils)
#reload(ev)
#reload(trainer)

import os
import glob
import datetime
import pandas as pd

t = {}

for f in glob.glob('autoencode/data/summary/*.summary'):
    t[f.split('/')[-1].replace('.summary', '')] = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    
t = pd.DataFrame(t.items(), columns=['name', 'time'])
t = t[t.name.str.startswith('hlf_eflow3_7_v')].sort_values('time')

utils.summary().filename
import glob
import pandas as pd

l = {}
for f in glob.glob('TEST/*'):
    lp = pd.read_csv(f)
    fp = f.split('/')[-1]
    lp['name'] = fp
    l[fp] = lp

l0 = pd.DataFrame()

if l:
    l = pd.concat(l)
    l['mass_nu_ratio'] = zip(l.mass, l.nu)
    l0 = l.pivot('mass_nu_ratio', 'name', 'auc')

s = utils.summary()
# l = l.drop(['Unnamed: 0', 'mass', 'nu'], axis=1).T

