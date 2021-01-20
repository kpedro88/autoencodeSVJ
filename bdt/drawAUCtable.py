import sys
sys.path.append("../training")

from module.DataLoader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import pandas as pd

qcd_path = "../../data/training_data/qcd/base_3/*.h5"
aucs_file_name = "trainingResults/aucs.txt"

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def read_aucs_file():
    
    aucs_dict = {}
    
    with open(aucs_file_name, "r") as aucs_file:
        lines = aucs_file.readlines()
        
        for line in lines:
            (mass, rinv, auc) = line.split("\t")
            mass = float(mass)
            rinv = float(rinv)
            auc = float(auc)
            
            aucs_dict[(mass, rinv)] = auc

    return aucs_dict


def plot_signal_aucs_from_lp(lp, auc_dict, title=None):
    fac = 1.5
    
    plt.figure(figsize=(1.1 * fac * 6.9, 1.1 * fac * 6))
    plt.imshow(lp, cmap='viridis')
    
    cb = plt.colorbar()
    cb.set_label(label='AUC value', fontsize=18 * fac)
    
    # plt.xticks(np.arange(0, 5, 1), map(lambda x: '{:.2f}'.format(x), np.unique(lp.columns)))
    plt.yticks(np.arange(0, 6, 1), np.unique(lp.index))
    
    plt.title(title, fontsize=fac * 25)
    plt.ylabel(r'$M_{Z^\prime}$ (GeV)', fontsize=fac * 20)
    plt.xlabel(r'$r_{inv}$', fontsize=fac * 20)
    plt.xticks(fontsize=18 * fac)
    plt.yticks(fontsize=18 * fac)

    for mi, (mass, row) in enumerate(lp.iterrows()):
        for ni, (nu, auc) in enumerate(row.iteritems()):
            plt.text(ni, mi, '{:.3f}'.format(auc), ha="center", va="center", color="w", fontsize=18 * fac)

    # for (mass, rinv), auc in aucs_dict.items():
    #
    #     plt.text(rinv, mass, '{:.3f}'.format(aucs_dict[(mass, rinv)]),
    #              ha="center", va="center", color="w", fontsize=18 * fac)
    

aucs_dict = read_aucs_file()

print("auc dict: ", aucs_dict)


aucs_dict_2 = {}

first_line = True

masses = set()

for (mass, rinv), auc in aucs_dict.items():
    if rinv in aucs_dict_2.keys():
        aucs_dict_2[rinv].append(auc)
    else:
        aucs_dict_2[rinv] = [auc]
        
    masses.add(mass)

massesSorted = sorted(masses)

aucs_dict_2["mass"] = massesSorted

print("new auc dict:", aucs_dict_2)

df = pd.DataFrame(aucs_dict_2)

df.set_index('mass', inplace = True,  append = True, drop = True)

# df.set_index('mass')

print("dataframe: ", df)

plot_signal_aucs_from_lp(df, aucs_dict)

plt.show()



# decisions = bdt.decision_function(X_test)
# # Compute ROC curve and area under the curve
# fpr, tpr, thresholds = roc_curve(Y_test, decisions)
# roc_auc = auc(fpr, tpr)
#
# plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
#
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.grid()
# plt.show()
