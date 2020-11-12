import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import pandas as pd

# ------------------------------------------------------------------------------------------------
# This script will draw input and reconstructed variables for signal found in the
# "signals_base_path" and background as specified in the training summary.
# ------------------------------------------------------------------------------------------------

scaler_type = "customStandardScaler"

training_version = {"standardScaler": 8,
                    "customScaler": 47,
                    "robustScaler": 63,
                    "customStandardScaler": 86,
                    "": None
                    }

summaries_path = "trainingResults/summary/{}/".format(scaler_type)


efp_base = 3
bottleneck_dim = 8
summary_base_name = "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim)

input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=summaries_path,
                                                                   file_name_base=summary_base_name,
                                                                   version=training_version[scaler_type])
    
print("Loading summary: ",          input_summary_path             )

# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2500]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]

signals_base_path = "../../data/training_data/all_signals/"

signals = {"signal_{}_{}".format(mass, rinv).replace(".", "p") : "{}{}GeV_{:1.2f}/base_3/*.h5".format(signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)


n_columns = 5
n_rows = 4

canvas = plt.figure()

i_plot = 1

bins = {
    "Eta" : (-3.5, 3.5, 100),
    "Phi" : (-3.5, 3.5, 100),
    "Pt"  : (0, 2000, 100),
    "M"  : (0, 800, 100),
    "ChargedFraction"  : (0, 1, 100),
    "PTD"  : (0, 1, 100),
    "Axis2"  : (0, 0.2, 100),
    # "Flavor"  : (0, 1000, 100),
    # "Energy"  : (0, 1000, 100),
    "eflow 1"  : (0, 1, 100),
    "eflow 2"  : (0, 1, 100),
    "eflow 3"  : (0, 1, 100),
    "eflow 4"  : (0, 1, 100),
    "eflow 5"  : (0, 1, 100),
    "eflow 6"  : (0, 1, 100),
    "eflow 7"  : (0, 1, 100),
    "eflow 8"  : (0, 1, 100),
    "eflow 9"  : (0, 1, 100),
    "eflow 10"  : (0, 1, 100),
    "eflow 11"  : (0, 1, 100),
    "eflow 12"  : (0, 1, 100),
}

def draw_histogram_for_variable(input_data, reconstructed_data, variable_name, i_plot):
    hist = canvas.add_subplot(n_rows, n_columns, i_plot)
    
    input_values = input_data[variable_name]
    reconstructed_values = reconstructed_data[variable_name]
    
    hist.hist(input_values, bins=numpy.linspace(*bins[variable_name]), alpha=0.5, label='input', histtype="step", density=True)
    hist.hist(reconstructed_values, bins=numpy.linspace(*bins[variable_name]), alpha=0.5, label='reconstruction', histtype="step", density=True)
    hist.title.set_text(variable_name)


print("input test data size: ", len(evaluator.qcd_test_data.df.index))
print("input normalized test data size: ", len(evaluator.qcd_test_data_normalized.df.index))
print("reco normalized data size: ", len(evaluator.qcd_recon.df.index))

for variable_name in bins:
    draw_histogram_for_variable(input_data=evaluator.qcd_test_data,
                                reconstructed_data=evaluator.qcd_recon,
                                variable_name=variable_name, i_plot=i_plot)
    i_plot += 1




legend = canvas.add_subplot(n_rows, n_columns, i_plot)
legend.hist([], alpha=0.5, label='input', histtype="step")
legend.hist([], alpha=0.5, label='reconstruction', histtype="step")
legend.legend(loc='upper right')

plt.show()