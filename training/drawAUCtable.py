import glob
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import module.SummaryProcessor as summaryProcessor

# ------------------------------------------------------------------------------------------------
# This script will draw a table with AUCs (areas under ROC curve) values based on the CSV
# file stored in "AUCs_path" with version specified by "training_version".
# ------------------------------------------------------------------------------------------------

scaler_type = "customScaler"

training_version = {
    "standardScaler": 56,
    "customScaler": 12,
    "robustScaler": 47,
    "minMaxScaler": 57,
    "maxAbsScaler": 76,
    "customStandardScaler": 86,
}

efp_base = 3
bottleneck_dim = 8

AUCs_path = "trainingResults/aucs/{}/*".format(scaler_type)
summaries_path = "trainingResults/summary/{}/".format(scaler_type)

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_signal_aucs_from_lp(lp, title=None):
    fac = 1.5
    
    plt.figure(figsize=(1.1 * fac * 6.9, 1.1 * fac * 6))
    plt.imshow(lp, cmap='viridis')
    
    cb = plt.colorbar()
    cb.set_label(label='AUC value', fontsize=18 * fac)
    
    plt.xticks(np.arange(0, 5, 1), map(lambda x: '{:.2f}'.format(x), np.unique(lp.columns)))
    plt.yticks(np.arange(0, 6, 1), np.unique(lp.index))
    
    plt.title(title, fontsize=fac * 25)
    plt.ylabel(r'$M_{Z^\prime}$ (GeV)', fontsize=fac * 20)
    plt.xlabel(r'$r_{inv}$', fontsize=fac * 20)
    plt.xticks(fontsize=18 * fac)
    plt.yticks(fontsize=18 * fac)
    
    for mi, (mass, row) in enumerate(lp.iterrows()):
        for ni, (nu, auc) in enumerate(row.iteritems()):
            plt.text(ni, mi, '{:.3f}'.format(auc), ha="center", va="center", color="w", fontsize=18 * fac)
    
    return plt.gca()


def plot_signal_aucs(aucs, title=None):
    
    lp = aucs.iloc[:, np.argsort(aucs.mean()).values[::-1][:1]].mean(axis=1).to_frame().reset_index().rename(columns={0: 'auc'})
    
    lp['mass'] = lp.mass_nu_ratio.apply(lambda x: x[0])
    lp['nu'] = lp.mass_nu_ratio.apply(lambda x: x[1])
    
    lp = lp.drop('mass_nu_ratio', axis=1).pivot('mass', 'nu', 'auc')
    
    print("lp type:" , type(lp))
    print("lp: ", lp)
    
    
    return lp, plot_signal_aucs_from_lp(lp, title)


auc_dict = {}

for f in glob.glob(AUCs_path):
    data_elt = pd.read_csv(f)
    file_elt = str(f.split('/')[-1])
    data_elt['name'] = file_elt
    auc_dict[file_elt] = data_elt
    
print("AUC dict: ", auc_dict)

aucs = pd.concat(auc_dict)
aucs['mass_nu_ratio'] = list(zip(aucs.mass, aucs.nu))
aucs = aucs.pivot('mass_nu_ratio', 'name', 'auc')



summaries = summaryProcessor.summary(summary_path=summaries_path)

model_acceptance_fraction = 10  # take top N best performing models
# take lowest 10% losses of all trainings
# n_best = int(0.01 * model_acceptance_fraction * summaries.data.size)
# best_ = summaries.sort_values('total_loss').head(n_best)
# best_name = str(best_.filename.values[0])
#
# print("N best: ", n_best)
# print("Best models: ", best_)
# print("The best model: ", best_name)



AUC_file_name = "hlf_eflow{}_{}_v{}".format(efp_base, bottleneck_dim, training_version[scaler_type])

print("AUCs: ", aucs[AUC_file_name].to_frame())

best, ax = plot_signal_aucs(aucs[AUC_file_name].to_frame(), title='Autoencoder AUCs (Best AE)')

plt.show()