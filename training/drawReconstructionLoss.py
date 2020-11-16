import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import pandas as pd

# ------------------------------------------------------------------------------------------------
# This script will draw reconstruction loss for a a mixture of all models found in the
# "signals_base_path" and backgrounds as specified in the training summary file.
# ------------------------------------------------------------------------------------------------

scalers_and_best_training_versions = {"standardScaler": 56,
                                      "customScaler": 12,
                                      "robustScaler": 47,
                                      "minMaxScaler": 57,
                                      "maxAbsScaler": 76,
                                      "customStandardScaler": 86,
                                      }

signals_base_path = "../../data/training_data/all_signals/"

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2500]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]

efp_base = 3
bottleneck_dim = 8

def get_signals():
    signals = {"signal_{}_{}".format(mass, rinv).replace(".", "p"):
                   "{}{}GeV_{:1.2f}/base_3/*.h5".format(signals_base_path, mass, rinv)
               for mass in masses
               for rinv in rinvs}
    
    return signals


def get_evaluator(scaler_type, training_version):
    summaries_path = "trainingResults/summary/{}/".format(scaler_type)
    summary_base_name = "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim)
    input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=summaries_path,
                                                                       file_name_base=summary_base_name,
                                                                       version=training_version)
    
    signals = get_signals()
    return AutoEncoderEvaluator(input_summary_path, signals=signals)


def get_losses(scaler_type, training_version):
    evaluator = get_evaluator(scaler_type, training_version)
    
    loss_qcd = evaluator.qcd_err.mae
    loss_signal = []
    
    for signal in get_signals():
        signal_mae_array = getattr(evaluator, "{}_err".format(signal)).mae
        loss_signal.append(signal_mae_array)
    
    loss_signal = pd.concat(loss_signal)
    
    return loss_qcd, loss_signal


n_columns = 2
n_rows = 2

canvas = plt.figure(figsize=(10, 10))
i_plot = 1

for scaler_type, training_version in scalers_and_best_training_versions.items():

    loss_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
    i_plot += 1
    
    loss_qcd, loss_signal = get_losses(scaler_type, training_version)
    
    loss_hist.hist(loss_qcd, bins=numpy.linspace(0, 0.4, 100), label="qcd", histtype="step", density=True)
    loss_hist.hist(loss_signal, bins=numpy.linspace(0, 0.4, 100), label="signal", histtype="step", density=True)
    loss_hist.set_yscale("log")
    loss_hist.set_ylim(bottom=1E-2, top=1E2)
    loss_hist.title.set_text(scaler_type)
    loss_hist.legend(loc='upper right')

plt.show()