import module.utils as utils
import module.evaluate as ev
import module.trainer as trainer
import module.summaryProcessor as summaryProcessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
import glob

print("imports OK")

output_path = "trainingResults/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../../data/training_data/"
signal_path = input_path + "all_signals/2000GeV_0.15/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"


def loadSummaries():
    print("\nRunning loadSummaries\n")
    summary = summaryProcessor.summary(summary_path)
    summaryWithOutdated = summaryProcessor.summary(summary_path=summary_path, include_outdated=True)
    return summary, summaryWithOutdated

summary, summaryWithOutdated = loadSummaries()

# aucs = ev.load_auc_table(path="autoencode/data/aucs")

# print(aucs.mean().sort_values()[::-1])

# consider = summary.sort_values('start_time')[::-1].iloc[1:15].filename
# a = aucs.loc[:,consider.values]




def printSummary():
    print("\nRunning printSummary\n")
    if summary is None:
        print("Could not print summary")
        return
    print("Full summary:", summary[summary.filename == 'hlf_eflow3_5_v0'].T)
    print("Summary - mae_auc: ", summary.sort_values('mae_auc')[::-1].T)

def plotAuc():
    """ Some plotting code """
    print("\nRunning plotAuc\n")
    if summary is None:
        print("Could not load summary for plotting")
        return
    
    res = summary[pd.DatetimeIndex(summary.start_time) > datetime.datetime(year=2020, month=6, day=1, hour=22, minute=30)]
    res = res[res.epochs == 100]
    
    plt.scatter(res.total_loss, res.mae_auc)
    plt.scatter(res.total_loss, res.mse_auc)
    plt.xlabel("Total loss")
    plt.ylabel("AUC (mae)")
    plt.show()


def printSummaries():
    """ Some code printing summaries and their timestamps """
    print("\nRunning printSummaries\n")
    summaries_timestamps = {}

    for filePath in glob.glob(summary_path+"/*.summary"):
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

def trainWithRandomRate(n_trainings, bottleneck_dim, batch_size):
    print("\nRunning trainWithRandomRate\n")

    mu, sigma = 0.00075, 0.01

    for i in range(n_trainings):
        print("\n\nStarting training ", i, "\n\n")
        learning_rate = np.random.lognormal(np.log(mu), sigma)
        print('target_dim {}, learning_rate {}:'.format(bottleneck_dim, learning_rate),)
        auc = ev.ae_train(
            signal_path=signal_path,
            qcd_path=qcd_path,
            output_data_path=output_path,
            target_dim=bottleneck_dim,
            verbose=False,
            batch_size=batch_size,
            learning_rate=learning_rate,
            norm_percentile=25
        )
        total_loss, ae, test_norm = auc
        print("Total loss: ", total_loss)

def trainAndEvaluate(bottleneck_dim, batch_size, learning_rate, epochs, es_patience, lr_patience, lr_factor, norm_percentile):
    print("\nRunning trainAndEvaluate\n")
    auc = ev.ae_train(
        signal_path=signal_path,
        qcd_path=qcd_path,
        output_data_path=output_path,
        target_dim=bottleneck_dim,
        verbose=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        es_patience=es_patience,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        norm_percentile=norm_percentile
    )

    total_loss, ae, test_norm = auc

    print("Total loss: ", total_loss)
    print("Autoencoder: ", ae)
    print("Test norm: ", test_norm)

    ae.evaluate(test_norm.values, test_norm.values)
    

def updataSignalEvals():
    print("\nRunning updateSignalEvals\n")
    ev.save_all_missing_AUCs(background_path=qcd_path,
                             signals_path=signal_path,
                             output_path=output_path)


def get_training_info_dict(filepath):
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    if not os.path.exists(filepath):
        print("Could not open file: ", filepath)
        raise AttributeError
    return trainer.pkl_file(filepath).store.copy()

def printTrainingInfo():
    print("\nRunning printTrainingInfo\n")
    trainingOutputPath = results_path+"/hlf_eflow3_5_v0.pkl"
    trainingInfo = get_training_info_dict(trainingOutputPath)
    print("Training info - val_loss: ", trainingInfo['metrics']['val_loss'][-1,1])


def getLatestSummaryFilePath(efp_base, bottleneck_dim, version=None):
    if version is None:
        version = summaryProcessor.get_last_summary_file_version(output_path, "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim))

    input_summary_path = summary_path+"/hlf_eflow{}_{}_v{}.summary".format(efp_base, bottleneck_dim, version)
    return input_summary_path
    

def drawROCcurve(efp_base, bottleneck_dim, version=None):
    input_summary_path = getLatestSummaryFilePath(efp_base, bottleneck_dim, version)
    print("Drawing ROC curve for summary: ", input_summary_path)

    # masses = [1500, 2000, 2500, 3000, 3500, 4000]
    masses = [2000]
    rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
    base_path = "../../data/training_data/all_signals/"
    aux_signals = {"{}, {}".format(mass, rinv) : "{}{}GeV_{:1.2f}/base_3/*.h5".format(base_path, mass, rinv) for mass in masses for rinv in rinvs}

    # elt = ev.ae_evaluation(input_summary_path, qcd_path=qcd_path, SVJ_path=signal_path)
    elt = ev.ae_evaluation(input_summary_path, qcd_path=qcd_path, SVJ_path=signal_path, aux_signals_dict=aux_signals)
    elt.roc(xscale='log', metrics=["mae"])

def addSampleAndPrintShape():
    newData = utils.data_loader(name="data")
    newData.add_sample(input_path+"qcd/base_3/data_0_data.h5")
    
    print("shape: ", newData.data['jet_features'].shape)
    
    # print("shape: ", newData.data['jet_constituents'].shape)


# def consts(globpath):
#     files = glob.glob(globpath)
#     assert len(files) > 0, "no files in globpath '{}'".format(globpath)
#     l = utils.data_loader(name="data")
#     for f in files:
#         l.add_sample(f)
#     stacked = np.vstack(l.data['jet_constituents'])
#     jets = np.vstack(l.data['jet_features'])
#     print(l.labels['jet_constituents'])
#
#     ret = []
#     tab = []
#     for i, elt in enumerate(stacked):
#         ret.append([])
#         tab.append(rt.TLorentzVector())
#         tab[i].SetPtEtaPhiE(jets[i][2], jets[i][0], jets[i][1], jets[i][8])
#         for j, const in enumerate(elt):
#             if const[4] > 0.0:
#                 ret[i].append(rt.TLorentzVector())
#                 ret[i][j].SetPtEtaPhiE(const[2], const[0], const[1], const[4])
#
#     return ret, tab
#
# data, jets = consts("TEST.h5")

def plotJetFeatures(efp_base, bottleneck_dim, version=None):
    input_summary_path = getLatestSummaryFilePath(efp_base, bottleneck_dim, version)
    
    evaluation = ev.ae_evaluation(input_summary_path)
    main = evaluation.qcd
    
    for normer, rng in [
        ({'norm_type': 'MinMaxScaler', 'feature_range': (0, 1)}, (0, 1)),
        ({'norm_type': 'StandardScaler'}, (-5., 5.)),
        ({'norm_type': 'RobustScaler'}, (-5., 5.))
    ]:
        var = main.cdrop('eflow *').normalize(**normer)
        
        plt.figure(figsize=(9, 9))
        for colname in var.axes[1]:
            plt.hist(var[colname], bins=70, range=rng, label=colname, histtype='step')
        plt.legend()
        plt.show()



def loadAndPrintQCD():
    data, jets, event, flavor = utils.load_all_data(qcd_path, name='QCD')
    # print(help(ef.datasets.qg_jets.load))
    
    print("data: ", data)
    print("jets: ", jets)
    print("event: ", event)
    print("flavor: ", flavor)





# lr = .00051
# es_patience = 12
# norm_percentile = 25
# epochs = 100

# trainAndEvaluate(bottleneck_dim=8,
#                  batch_size=32,
#                  learning_rate=0.00051,
#                  epochs=10,
#                  es_patience=12,
#                  lr_patience=9,
#                  lr_factor=0.5,
#                  norm_percentile=25
#                  )

# printSummary()
# plotAuc()
drawROCcurve(efp_base=3, bottleneck_dim=8, version=24)
# addSampleAndPrintShape()
# plotJetFeatures(efp_base=3, bottleneck_dim=8)
# trainWithRandomRate(n_trainings=10, bottleneck_dim=8, batch_size=64)
# loadAndPrintQCD()

# ev.update_all_signal_evals(summary_path=summary_path, signal_path=signal_path, qcd_path=qcd_path, path="trainingResults/aucs")


def get_signal_auc_df(aucs, n_avg=1, do_max=False):
    lp = None
    if do_max:
        lp = aucs.max(axis=1).to_frame().reset_index().rename(columns={0: 'auc'})
    else:
        lp = aucs.iloc[
             :, np.argsort(aucs.mean()).values[::-1][:n_avg]
             ].mean(axis=1).to_frame().reset_index().rename(columns={0: 'auc'})

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

# aucs = ev.load_auc_table("trainingResults/aucs")
#
# best_name = "hlf_eflow3_8_v0"
#
# best,ax = plot_signal_aucs(aucs[best_name].to_frame(), title='Autoencoder AUCs (Best AE)')
#
#
#
# plt.show()












