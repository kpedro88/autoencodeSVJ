import autoencode.module.autoencodeSVJ.utils as utils
import autoencode.module.autoencodeSVJ.trainer as trainer
import autoencode.module.autoencodeSVJ.evaluate as ev

import pandas as pd
import numpy as np

import datetime
import os
import glob

import matplotlib.pyplot as plt

print("imports OK")



# aucs = ev.load_auc_table(path="autoencode/data/aucs")
# s = utils.summary()
# consider = s.sort_values('start_time')[::-1].iloc[1:15].filename
# a = aucs.loc[:,consider.values]
# x = s[s.filename == 'hlf_eflow3_8_v36'].T
# print(x)
# s = utils.summary(include_outdated=True)
# print(aucs.mean().sort_values()[::-1])

# res = utils.summary()
# res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime(year=2020, month=6, day=1, hour=22, minute=30)]
# res = res[res.epochs == 100]

# plt.scatter(res.total_loss, res.mae_auc)
# plt.scatter(res.total_loss, res.mse_auc)
# plt.show()


# t = {}
#
# for f in glob.glob('autoencode/data/summary/*.summary'):
#     t[f.split('/')[-1].replace('.summary', '')] = datetime.datetime.fromtimestamp(os.path.getmtime(f))
#
# t = pd.DataFrame(t.items(), columns=['name', 'time'])
# t = t[t.name.str.startswith('hlf_eflow3_7_v')].sort_values('time')
#
# print(utils.summary().filename)
#
#
# l = {}
# for f in glob.glob('TEST/*'):
#     lp = pd.read_csv(f)
#     fp = f.split('/')[-1]
#     lp['name'] = fp
#     l[fp] = lp
#
# l0 = pd.DataFrame()
#
# if l:
#     l = pd.concat(l)
#     l['mass_nu_ratio'] = zip(l.mass, l.nu)
#     l0 = l.pivot('mass_nu_ratio', 'name', 'auc')
#
# s = utils.summary()
# # l = l.drop(['Unnamed: 0', 'mass', 'nu'], axis=1).T
#
# #help(ev.ae_train)
#
# signal_dir = '../data/all_signals'
# signals = glob.glob('{}/*'.format(signal_dir))
# dim = 7
# # for i in range(100):
# #     mu, sigma = 0.00075, 0.01
# #     lr = np.random.lognormal(np.log(mu), sigma)
# #     print('target_dim {}, lr {}:'.format(dim, lr),)
# #     auc = ev.ae_train(
# #         signal_path='../data/all_signals/1500GeV_0p75/*.h5',
# #         qcd_path='../data/qcd/*.h5',
# #         target_dim=dim,
# #         verbose=False,
# #         batch_size=32,
# #         learning_rate=lr,
# #         norm_percentile=25
# #     )
# #     print(auc)
#

auc = ev.ae_train(
    signal_path='../data/all_signals/1500GeV_0p75/base_3/*.h5',
    qcd_path='../data/qcd/*.h5',
    output_data_path="trainingResults",
    target_dim=5,
    verbose=1,
    batch_size=32,
    learning_rate=0.001,
    epochs=5,
)

print("Running update_all_signal_evals")
ev.update_all_signal_evals(background_path="../data/qcd/*.h5",
                           signal_path="../data/all_signals/*",
                           output_path="trainingResults")

total_loss, ae, test_norm = auc

print("Total loss: ", total_loss)
print("Autoencoder: ", ae)
print("Test norm: ", test_norm)


#
# # help(ae.evaluate)
#
# print(ev.get_training_info_dict('hlf_eflow3_5_v21')['metrics']['val_loss'][-1,1])
#
# ae.evaluate(test_norm.values, test_norm.values)
#
# print(utils.summary().sort_values('mae_auc')[::-1].T)
#
# # help(ev.ae_train)
#
# for i in range(20):
#     mu, sigma = 0.002, 0.925
#     lr = np.random.lognormal(np.log(mu), sigma)
#     print('target_dim {}, lr {}:'.format(dim, lr))
#     print('  aucs: ',)
#     for i in range(5):
#         auc = ev.ae_train(
#             signal_path='../data/all_signals/1500GeV_0p75/base_3/*.h5',
#             qcd_path='../data/qcd/*.h5',
#             target_dim=dim,
#             verbose=False,
#             batch_size=32,
#             learning_rate=lr,
#         )
#         print(auc, ' ',)
#
