import time
import datetime
import module.utils as utils
import module.evaluate as ev
import module.summaryProcessor as summaryProcessor
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import OrderedDict as odict
import pandas as pd
import glob
import os
import tensorflow as tf

output_path = "trainingResults/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../../data/training_data/"
signal_path = input_path + "all_signals/2000GeV_0.75/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_signal_auc_df(aucs, n_avg=1, do_max=False):
    lp = None
    
    print("aucs: ", aucs)
    
    if do_max:
        lp = aucs.max(axis=1).to_frame().reset_index().rename(columns={0: 'auc'})
    else:
        lp = aucs.iloc[
             :, np.argsort(aucs.mean()).values[::-1][:n_avg]
             ].mean(axis=1).to_frame().reset_index().rename(columns={0: 'auc'})

    print("lp: ", lp)

    lp['mass'] = lp.mass_nu_ratio.apply(lambda x: x[0])
    lp['nu'] = lp.mass_nu_ratio.apply(lambda x: x[1])

    lp = lp.drop('mass_nu_ratio', axis=1).pivot('mass', 'nu', 'auc')

    return lp


def plot_signal_aucs_from_lp(lp, n_avg=1, do_max=False, title=None, fac=1.5, barlabel=None, cmap='viridis'):
    plt.figure(figsize=(1.1 * fac * 6.9, 1.1 * fac * 6))

    plt.imshow(lp, cmap=cmap)
    if barlabel == None:
        barlabel = 'AUC value'
    cb = plt.colorbar()
    cb.set_label(label=barlabel, fontsize=18 * fac)

    plt.xticks(np.arange(0, 5, 1), map(lambda x: '{:.2f}'.format(x), np.unique(lp.columns)))
    plt.yticks(np.arange(0, 6, 1), np.unique(lp.index))

    if title is not None:
        plt.title(title, fontsize=fac * 25)
    elif do_max:
        plt.title('Best AUCs (for any autoencoder)', fontsize=fac * 25)
    elif n_avg < 2:
        plt.title('Signal AUCs (best autoencoder)', fontsize=fac * 25)
    else:
        plt.title('Average Signal AUCs (best {} models)'.format(n_avg), fontsize=fac * 25)
    plt.ylabel(r'$M_{Z^\prime}$ (GeV)', fontsize=fac * 20)
    plt.xlabel(r'$r_{inv}$', fontsize=fac * 20)
    plt.xticks(fontsize=18 * fac)
    plt.yticks(fontsize=18 * fac)

    for mi, (mass, row) in enumerate(lp.iterrows()):
        for ni, (nu, auc) in enumerate(row.iteritems()):
            plt.text(ni, mi, '{:.3f}'.format(auc), ha="center", va="center", color="w", fontsize=18 * fac)

    return plt.gca()


def plot_signal_aucs(aucs, n_avg=1, do_max=False, title=None, fac=1.5, cmap='viridis'):
    lp = get_signal_auc_df(aucs, n_avg, do_max)
    return lp, plot_signal_aucs_from_lp(lp, n_avg, do_max, title, fac, cmap=cmap)


TRAIN = True

lr = .00051
lr_factor = 0.5
es_patience = 12
target_dim = 8
batch_size = 32
norm_percentile = 25
epochs = 2
n_models = 10  # number of models to train
model_acceptance_fraction = 10  # take top N best performing models

start_stamp = time.time()
res = None
if TRAIN:
    for i in range(n_models):
        mse = ev.ae_train(
            output_data_path=output_path,
            signal_path=signal_path,
            qcd_path=qcd_path,
            target_dim=target_dim,
            verbose=False,
            batch_size=batch_size,
            learning_rate=lr,
            norm_percentile=norm_percentile,
            lr_factor=lr_factor,
            es_patience=es_patience,
            epochs=epochs
        )
        print('model {} finished'.format(i))

    res = summaryProcessor.summary(summary_path=summary_path)
    res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime.fromtimestamp(start_stamp)]

else:
    res = summaryProcessor.summary(summary_path=summary_path)
    res = res.sort_values('start_time').tail(n_models)

print("res: ", res)

# take lowest 10% losses of all trainings
n_best = int(0.01 * model_acceptance_fraction * n_models)
best_ = res.sort_values('total_loss').head(n_best)
best_name = str(best_.filename.values[0])

print("N best: ", n_best)
print("Best models: ", best_)
print("The best model: ", best_name)

ev.update_all_signal_evals(summary_path=summary_path,
                           path=output_path+"/aucs",
                           qcd_path=qcd_path,
                           signal_path=(input_path + "all_signals/*/base_3/*.h5"))

aucs = ev.load_auc_table("trainingResults/aucs")
# bdts = pd.read_csv(output_path+"/bdt_aucs.csv")
# bdts = bdts[bdts.columns[1:]].set_index(bdts[bdts.columns[0]].rename('mass'))
# bdts = bdts.T.set_index(bdts.T.index.rename('nu')).T
# bdts.columns = map(float, bdts.columns)
# bdts.index = map(float, bdts.index)


# plot_signal_aucs_from_lp(bdts, title='BDT AUCs (trained on each signal)')
best,ax = plot_signal_aucs(aucs[best_name].to_frame(), title='Autoencoder AUCs (Best AE)')
best,ax = plot_signal_aucs(aucs[best_.filename], title='Autoencoder AUCs\n(Average of Top 10\% of Models)', n_avg=len(best_))
best,ax = plot_signal_aucs(aucs[best_.filename], do_max=1, title='Autoencoder AUCs\n(Best of Top 10\% of Models)', n_avg=len(best_))

plt.show()