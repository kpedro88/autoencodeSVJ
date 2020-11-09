import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import pandas as pd

# ------------------------------------------------------------------------------------------------
# This script will draw ROC curves for a specified model against all signals found in the
# "signals_base_path". If no version is specified (set to None), the latest training
# will be used.
# ------------------------------------------------------------------------------------------------

training_version = 8
efp_base = 3
bottleneck_dim = 8
summaries_path = "trainingResults/summary/standardScaler/"
summary_base_name = "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim)

input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=summaries_path,
                                                                   file_name_base=summary_base_name,
                                                                   version=training_version)


# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2500]
# rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
rinvs = [0.45]

signals_base_path = "../../data/training_data/all_signals/"

signals = {"signal_{}_{}".format(mass, rinv).replace(".", "p") : "{}{}GeV_{:1.2f}/base_3/*.h5".format(signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

print("\n\nDraing ROC curves for summary: ", input_summary_path)

evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)


n_columns = 3
n_rows = 3

canvas = plt.figure()

i_plot = 1

eta_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

input_qcd_jet_eta = evaluator.qcd_test_data.Eta
reco_qcd_jet_eta = evaluator.qcd_recon.Eta
eta_hist.hist(input_qcd_jet_eta, bins=numpy.linspace(-4, 4, 100), alpha=0.5, label='input', histtype="step")
eta_hist.hist(reco_qcd_jet_eta, bins=numpy.linspace(-4, 4, 100), alpha=0.5, label='reconstruction', histtype="step")
eta_hist.title.set_text("Eta")
eta_hist.legend(loc='upper right')


phi_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

input_qcd_jet_phi = evaluator.qcd_test_data.Phi
reco_qcd_jet_phi = evaluator.qcd_recon.Phi
phi_hist.hist(input_qcd_jet_phi, bins=numpy.linspace(-4, 4, 100), alpha=0.5, label='input', histtype="step")
phi_hist.hist(reco_qcd_jet_phi, bins=numpy.linspace(-4, 4, 100), alpha=0.5, label='reconstruction', histtype="step")
phi_hist.title.set_text("Phi")
phi_hist.legend(loc='upper right')

loss_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

loss_qcd = evaluator.qcd_err.mae

loss_signal = []

for signal in signals:
    signal_mae_array = getattr(evaluator, "{}_err".format(signal)).mae
    # print("\n\nSignal mae type: ", type(signal_mae_array))
    # print("\n\nSignal mae: ", signal_mae_array)
    loss_signal.append(getattr(evaluator, "{}_err".format(signal)).mae)

loss_signal = pd.concat(loss_signal)

# print("\n\nQCD loss: ", loss_qcd)

loss_hist.hist(loss_qcd, bins=numpy.linspace(0, 0.4, 100), label="qcd", histtype="step", density=True)
loss_hist.hist(loss_signal, bins=numpy.linspace(0, 0.4, 100), label="signal", histtype="step", density=True)
loss_hist.set_yscale("log")
loss_hist.legend(loc='upper right')


# --------------------------------------------------------------------------------------------------
# fill number of SV jets for QCD

qcd_event_indices = evaluator.qcd_test_event.index

qcd_errors = evaluator.qcd_err.df
qcd_events = evaluator.qcd_test_event.df
qcd_jets = evaluator.qcd_test_data.df

qcd_n_svj_hist_data = []
qcd_mt_0_svj_hist_data = []
qcd_mt_1_svj_hist_data = []
qcd_mt_2_svj_hist_data = []
qcd_mt_gt2_svj_hist_data = []
svj_jet_cut = 0.037

print("\n\nFilling background histograms")

n_events_per_class = 10000
n_events = 0

for iEvent in qcd_event_indices:
    n_events += 1
    if n_events > n_events_per_class:
        break
    
    mt = qcd_events["MT"][iEvent]
    
    i_jet_1 = 2 * iEvent + 0
    i_jet_2 = 2 * iEvent + 1

    jet_1 = qcd_jets.loc[i_jet_1, :]
    jet_2 = qcd_jets.loc[i_jet_2, :]
    
    loss_jet_1 = qcd_errors["mae"].loc[i_jet_1]
    loss_jet_2 = qcd_errors["mae"].loc[i_jet_2]

    n_svj_jets = 0
    
    if loss_jet_1 > svj_jet_cut:
        n_svj_jets += 1
    
    if loss_jet_2 > svj_jet_cut:
        n_svj_jets += 1
    
    if n_svj_jets == 0:
        qcd_mt_0_svj_hist_data.append(mt)
    elif n_svj_jets == 1:
        qcd_mt_1_svj_hist_data.append(mt)
    elif n_svj_jets == 2:
        qcd_mt_2_svj_hist_data.append(mt)
    else:
        qcd_mt_gt2_svj_hist_data.append(mt)
        
    
    qcd_n_svj_hist_data.append(n_svj_jets)

# --------------------------------------------------------------------------------------------------
# fill number of SV jets for Signals

signals_n_svj_hist_data = []
signal_mt_0_svj_hist_data = []
signal_mt_1_svj_hist_data = []
signal_mt_2_svj_hist_data = []
signal_mt_gt2_svj_hist_data = []

print("\n\nFilling signal histograms")



for signal in signals:
    print("\tprocessing ", signal)
    
    signal_mae_array = getattr(evaluator, "{}_err".format(signal)).mae
    # print("\n\nSignal mae type: ", type(signal_mae_array))
    # print("\n\nSignal mae: ", signal_mae_array)
    loss_signal.append(getattr(evaluator, "{}_err".format(signal)).mae)
    
    signal_event_indices = getattr(evaluator, "{}_event".format(signal)).index

    qcd_errors = getattr(evaluator, "{}_err".format(signal)).df
    qcd_events = getattr(evaluator, "{}_event".format(signal)).df
    qcd_jets = getattr(evaluator, "{}".format(signal)).df

    n_events = 0

    for iEvent in signal_event_indices:
        n_events += 1
        if n_events > n_events_per_class:
            break
    
        mt = qcd_events["MT"][iEvent]
    
        i_jet_1 = 2 * iEvent + 0
        i_jet_2 = 2 * iEvent + 1
    
        jet_1 = qcd_jets.loc[i_jet_1, :]
        jet_2 = qcd_jets.loc[i_jet_2, :]
    
        loss_jet_1 = qcd_errors["mae"].loc[i_jet_1]
        loss_jet_2 = qcd_errors["mae"].loc[i_jet_2]
    
        n_svj_jets = 0
    
        if loss_jet_1 > svj_jet_cut:
            n_svj_jets += 1
    
        if loss_jet_2 > svj_jet_cut:
            n_svj_jets += 1

        if n_svj_jets == 0:
            signal_mt_0_svj_hist_data.append(mt)
        elif n_svj_jets == 1:
            signal_mt_1_svj_hist_data.append(mt)
        elif n_svj_jets == 2:
            signal_mt_2_svj_hist_data.append(mt)
        else:
            signal_mt_gt2_svj_hist_data.append(mt)
    
        signals_n_svj_hist_data.append(n_svj_jets)




n_svj_jets_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

n_svj_jets_hist.hist(qcd_n_svj_hist_data, bins=numpy.linspace(0, 5, 6), label="qcd", histtype="step", density=True)
n_svj_jets_hist.hist(signals_n_svj_hist_data, bins=numpy.linspace(0, 5, 6), label="signals", histtype="step", density=True)
n_svj_jets_hist.title.set_text("N SV Jets")
n_svj_jets_hist.legend(loc='upper right')



mt_0_svj_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

mt_0_svj_hist.hist(qcd_mt_0_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="qcd", histtype="step")
mt_0_svj_hist.hist(signal_mt_0_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="signal", histtype="step")
mt_0_svj_hist.title.set_text("M_T (0 SVJ)")
mt_0_svj_hist.legend(loc='upper right')

mt_1_svj_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

mt_1_svj_hist.hist(qcd_mt_1_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="qcd", histtype="step")
mt_1_svj_hist.hist(signal_mt_1_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="signal", histtype="step")
mt_1_svj_hist.title.set_text("M_T (1 SVJ)")
mt_1_svj_hist.legend(loc='upper right')

mt_2_svj_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

mt_2_svj_hist.hist(qcd_mt_2_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="qcd", histtype="step")
mt_2_svj_hist.hist(signal_mt_2_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="signal", histtype="step")
mt_2_svj_hist.title.set_text("M_T (2 SVJ)")
mt_2_svj_hist.legend(loc='upper right')

mt_gt2_svj_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

mt_gt2_svj_hist.hist(qcd_mt_gt2_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="qcd", histtype="step")
mt_gt2_svj_hist.hist(signal_mt_gt2_svj_hist_data, bins=numpy.linspace(0, 5000, 100), label="signal", histtype="step")
mt_gt2_svj_hist.title.set_text("M_T (>2 SVJ)")
mt_gt2_svj_hist.legend(loc='upper right')

plt.show()


