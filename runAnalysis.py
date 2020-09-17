import autoencode.module.autoencodeSVJ.utils as utils
import autoencode.module.autoencodeSVJ.trainer as trainer
import autoencode.module.autoencodeSVJ.evaluate as ev
import autoencode.module.autoencodeSVJ.summaryProcessor as summaryProcessor

import pandas as pd
import numpy as np

import datetime
import os
import glob

import matplotlib.pyplot as plt

print("imports OK")

output_path = "trainingResults"
summaryPath = output_path+"/summary"
results_path = output_path+"/trainingRuns"
signal_path = "../data/all_signals/1500GeV_0p75/base_3/*.h5"
qcd_path = "../data/qcd/*.h5"


def loadSummaries():
    summary = summaryProcessor.summary(summaryPath)
    summaryWithOutdated = summaryProcessor.summary(summaryPath=summaryPath, include_outdated=True)
    return summary, summaryWithOutdated

summary, summaryWithOutdated = loadSummaries()

# aucs = ev.load_auc_table(path="autoencode/data/aucs")

# print(aucs.mean().sort_values()[::-1])




# consider = summary.sort_values('start_time')[::-1].iloc[1:15].filename
# a = aucs.loc[:,consider.values]




def printSummary():
    if summary is None:
        print("Could not print summary")
        return
    print("Full summary:", summary[summary.filename == 'hlf_eflow3_5_v0'].T)
    print("Summary - mae_auc: ", summary.sort_values('mae_auc')[::-1].T)

def plotAuc():
    """ Some plotting code """
    
    if summary is None:
        print("Could not load summary for plotting")
        return
    
    res = summary[pd.DatetimeIndex(summary.start_time) > datetime.datetime(year=2020, month=6, day=1, hour=22, minute=30)]
    res = res[res.epochs == 100]
    plt.scatter(res.total_loss, res.mae_auc)
    plt.scatter(res.total_loss, res.mse_auc)
    plt.show()


def printSummaries():
    """ Some code printing summaries and their timestamps """
    summaries_timestamps = {}

    for filePath in glob.glob(summaryPath+"/*.summary"):
        file_name = filePath.split('/')[-1].replace(".summary", "")
        summaries_timestamps[file_name] = datetime.datetime.fromtimestamp(os.path.getmtime(filePath))

    print("summaries timestamps:\n", summaries_timestamps)

    t = pd.DataFrame(summaries_timestamps.items(), columns=['name', 'time'])
    t = t[t.name.str.startswith('hlf_eflow3_5_v')].sort_values('time')
    
    print("summaries timestamps processed:\n", t)


# l = {}
#
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
# l = l.drop(['Unnamed: 0', 'mass', 'nu'], axis=1).T

# #help(ev.ae_train)
#

def trainWithRandomRate(n_trainings):
    dim = 7

    for i in range(n_trainings):
        print("\n\nStarting training ", i, "\n\n")
        
        mu, sigma = 0.00075, 0.01
        learning_rate = np.random.lognormal(np.log(mu), sigma)
        print('target_dim {}, learning_rate {}:'.format(dim, learning_rate),)
        auc = ev.ae_train(
            signal_path=signal_path,
            qcd_path=qcd_path,
            output_data_path=output_path,
            target_dim=dim,
            verbose=False,
            batch_size=32,
            learning_rate=learning_rate,
            norm_percentile=25
        )
        total_loss, ae, test_norm = auc
        print("Total loss: ", total_loss)




# auc = ev.ae_train(
#     signal_path='../data/all_signals/1500GeV_0p75/base_3/*.h5',
#     qcd_path='../data/qcd/*.h5',
#     output_data_path="trainingResults",
#     target_dim=5,
#     verbose=1,
#     batch_size=32,
#     learning_rate=0.001,
#     epochs=5,
# )

# print("Running update_all_signal_evals")
# ev.update_all_signal_evals(background_path="../data/qcd/*.h5",
#                            signal_path="../data/all_signals/*",
#                            output_path="trainingResults")

# total_loss, ae, test_norm = auc
#
# print("Total loss: ", total_loss)
# print("Autoencoder: ", ae)
# print("Test norm: ", test_norm)


#
# # help(ae.evaluate)
#

def printTrainingInfo():
    trainingOutputPath = results_path+"/hlf_eflow3_5_v0.pkl"
    trainingInfo = ev.get_training_info_dict(trainingOutputPath)
    print("Training info - val_loss: ", trainingInfo['metrics']['val_loss'][-1,1])

# ae.evaluate(test_norm.values, test_norm.values)


#
# # help(ev.ae_train)
#
# for i in range(20):
#     mu, sigma = 0.002, 0.925
#     learning_rate = np.random.lognormal(np.log(mu), sigma)
#     print('target_dim {}, learning_rate {}:'.format(dim, learning_rate))
#     print('  aucs: ',)
#     for i in range(5):
#         auc = ev.ae_train(
#             signal_path='../data/all_signals/1500GeV_0p75/base_3/*.h5',
#             qcd_path='../data/qcd/*.h5',
#             target_dim=dim,
#             verbose=False,
#             batch_size=32,
#             learning_rate=learning_rate,
#         )
#         print(auc, ' ',)
#

printSummary()
plotAuc()

# trainWithRandomRate(10)