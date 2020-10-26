import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
from module.AutoEncoderTrainer import AutoEncoderTrainer

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

run_training = False
run_produce_aucs = False
run_plot_ROC_curves = True
run_plot_AUC_table = False

output_path = "trainingResults/"
aucs_path = output_path+"aucs/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../../data/training_data/"
signal_path = input_path + "all_signals/2000GeV_0.15/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"


def get_latest_summary_file_path(efp_base, bottleneck_dim, version=None):
    if version is None:
        version = summaryProcessor.get_last_summary_file_version(summary_path, "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim))

    input_summary_path = summary_path+"/hlf_eflow{}_{}_v{}.summary".format(efp_base, bottleneck_dim, version)
    return input_summary_path




# ---------------------------------------------------------------------------------------------------------------------
# Run the training
#
if run_training:
    training_params = {
        'batch_size': 32,
        'loss': 'mse',
        'optimizer': 'adam',
        'epochs': 2,
        'learning_rate': 0.00051,
        'es_patience': 12,
        'lr_patience': 9,
        'lr_factor': 0.5
    }
    
    trainer = AutoEncoderTrainer(qcd_path=qcd_path,
                                 bottleneck_size=8,
                                 training_params=training_params,
                                 norm_percentile=25
                                 )
    
    trainer.run_training(training_output_path=results_path,
                         summaries_path=summary_path,
                         verbose=True
                         )
    
    trainer.save_last_training_summary(path=summary_path)




# ---------------------------------------------------------------------------------------------------------------------
# Translate summaries to AUC files
#

if run_produce_aucs:
    summaryProcessor.save_all_missing_AUCs(summary_path=summary_path,
                                           signals_path=(input_path + "all_signals/*/base_3/*.h5"),
                                           AUCs_path=aucs_path)



# ---------------------------------------------------------------------------------------------------------------------
# Plot ROC curves
#

if run_plot_ROC_curves:

    input_summary_path = get_latest_summary_file_path(efp_base=3, bottleneck_dim=8)
    # masses = [1500, 2000, 2500, 3000, 3500, 4000]
    masses = [2000]
    rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
    base_path = "../../data/training_data/all_signals/"
    signals = {"{}, {}".format(mass, rinv): "{}{}GeV_{:1.2f}/base_3/*.h5".format(base_path, mass, rinv) for mass in masses for rinv in rinvs}
    
    elt = AutoEncoderEvaluator(input_summary_path, qcd_path=qcd_path, signals=signals)
    elt.roc(xscale='log', metrics=["mae"])





# ---------------------------------------------------------------------------------------------------------------------
# Plot AUC tables
#


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
    
    
    return lp, plot_signal_aucs_from_lp(lp, title)


if run_plot_AUC_table:
    auc_dict = {}
    
    for f in glob.glob(aucs_path+"*"):
        data_elt = pd.read_csv(f)
        file_elt = str(f.split('/')[-1])
        data_elt['name'] = file_elt
        auc_dict[file_elt] = data_elt
    
    aucs = pd.concat(auc_dict)
    aucs['mass_nu_ratio'] = list(zip(aucs.mass, aucs.nu))
    aucs = aucs.pivot('mass_nu_ratio', 'name', 'auc')
    
    print("aucs:", aucs)
    
    best_name = "hlf_eflow3_8_v0"
    best,ax = plot_signal_aucs(aucs[best_name].to_frame(), title='Autoencoder AUCs (Best AE)')
    
    
    plt.show()