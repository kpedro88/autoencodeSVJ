import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator
import matplotlib.pyplot as plt
import numpy
import ROOT

# ------------------------------------------------------------------------------------------------
# This script will draw ROC curves for a specified model against all signals found in the
# "signals_base_path". If no version is specified (set to None), the latest training
# will be used.
# ------------------------------------------------------------------------------------------------

scaler_type = "customStandardScaler"

training_version = {"standardScaler": 56,
                    "customScaler": 47,
                    "robustScaler": 63,
                    "customStandardScaler": 86
                    }

efp_base = 3
bottleneck_dim = 8
svj_jet_cut = 0.037
n_events_per_class = 10000

summaries_path = "trainingResults/summary/{}/".format(scaler_type)
summary_base_name = "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim)

input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=summaries_path,
                                                                   file_name_base=summary_base_name,
                                                                   version=training_version[scaler_type])


masses = [1500, 2000, 2500, 3000, 3500, 4000]
# masses = [2500]
rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]
# rinvs = [0.45]

signals_base_path = "../../data/training_data/all_signals/"

signals = {"mZprime{}_mDark20_rinv{}_alphapeak".format(mass, rinv).replace(".", "") : "{}{}GeV_{:1.2f}/base_3/*.h5".format(signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

def get_hist_data(events, event_indices, jets, errors):
    n_events = 0
    n_svj_hist_data = []
    mt_svj_hist_data = {0: [], 1: [], 2: []}
    
    for iEvent in event_indices:
        n_events += 1
        if n_events > n_events_per_class:
            break
        
        mt = events["MT"][iEvent]
        
        i_jet_1 = 2 * iEvent + 0
        i_jet_2 = 2 * iEvent + 1
        
        jet_1 = jets.loc[i_jet_1, :]
        jet_2 = jets.loc[i_jet_2, :]
        
        loss_jet_1 = errors["mae"].loc[i_jet_1]
        loss_jet_2 = errors["mae"].loc[i_jet_2]
        
        n_svj_jets = 0
        
        if loss_jet_1 > svj_jet_cut:
            n_svj_jets += 1
        
        if loss_jet_2 > svj_jet_cut:
            n_svj_jets += 1
        
        n_svj_hist_data.append(n_svj_jets)
        mt_svj_hist_data[n_svj_jets].append(mt)
        
    return n_svj_hist_data, mt_svj_hist_data


n_columns = 2
n_rows = 2
i_plot = 1

canvas = plt.figure(figsize=(10, 10))

evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)

# --------------------------------------------------------------------------------------------------
# fill number of SV jets for QCD

qcd_event_indices = evaluator.qcd_test_event.index
qcd_errors = evaluator.qcd_err.df
qcd_events = evaluator.qcd_test_event.df
qcd_jets = evaluator.qcd_test_data.df

print("\n\nFilling background histograms")
qcd_n_svj_hist_data, qcd_mt_svj_hist_data = get_hist_data(qcd_events, qcd_event_indices, qcd_jets, qcd_errors)

n_svj_hist = canvas.add_subplot(n_rows, n_columns, i_plot)
i_plot += 1

n_svj_hist.hist(qcd_n_svj_hist_data, bins=numpy.linspace(0, 5, 6), label="qcd", histtype="step", density=True)
n_svj_hist.title.set_text("N SV Jets")


mt_svj_hist = {}

output_file = ROOT.TFile("stat_hists.root", "recreate")
output_file.cd()



for n_svj in qcd_mt_svj_hist_data:
    mt_svj_hist[n_svj] = canvas.add_subplot(n_rows, n_columns, i_plot)
    i_plot += 1
    
    mt_svj_hist[n_svj].hist(qcd_mt_svj_hist_data[n_svj],
                            bins=numpy.linspace(0, 5000, 100), label="qcd", histtype="step")
    mt_svj_hist[n_svj].title.set_text("M_T ({} SVJ)".format(n_svj))

    mt_svj_root_hist = ROOT.TH1D("QCD", "QCD", 750, 0, 7500)
    
    for data in qcd_mt_svj_hist_data[n_svj]:
        mt_svj_root_hist.Fill(data)
        
    output_file.mkdir("SVJ{}_2018".format(n_svj))
    output_file.cd("SVJ{}_2018".format(n_svj))
    mt_svj_root_hist.Write()

    mt_svj_root_hist = mt_svj_root_hist.Clone()
    mt_svj_root_hist.SetName("Bkg")
    mt_svj_root_hist.Write()

    mt_svj_root_hist = mt_svj_root_hist.Clone()
    mt_svj_root_hist.SetName("data_obs")
    mt_svj_root_hist.Write()
    
    



    

# --------------------------------------------------------------------------------------------------
# fill number of SV jets for Signals

signals_n_svj_hist_data = []
signals_mt_svj_hist_data = {0: [], 1: [], 2: []}

print("\n\nFilling signal histograms")

for signal in signals:
    print("\tprocessing ", signal)
    
    signal_event_indices = getattr(evaluator, "{}_event".format(signal)).index
    signal_errors = getattr(evaluator, "{}_err".format(signal)).df
    signal_events = getattr(evaluator, "{}_event".format(signal)).df
    signal_jets = getattr(evaluator, "{}".format(signal)).df

    n_events = 0

    signal_n_svj_hist_data, signal_mt_svj_hist_data = get_hist_data(signal_events,
                                                                    signal_event_indices,
                                                                    signal_jets,
                                                                    signal_errors)

    signals_n_svj_hist_data.append(signal_n_svj_hist_data)
    
    for n_svj, data in signal_mt_svj_hist_data.items():
        signals_mt_svj_hist_data[n_svj].append(data)

        name = "SVJ_"+signal
        mt_svj_root_hist = ROOT.TH1D(name, name, 750, 0, 7500)
    
        for value in data:
            mt_svj_root_hist.Fill(value)
    
        output_file.cd("SVJ{}_2018".format(n_svj))
        mt_svj_root_hist.Write()


n_svj_hist.hist(signals_n_svj_hist_data, bins=numpy.linspace(0, 5, 6), label="signals", histtype="step", density=True)
n_svj_hist.legend(loc='upper right')

for n_svj, data in signals_mt_svj_hist_data.items():
    mt_svj_hist[n_svj].hist(data, bins=numpy.linspace(0, 5000, 100), label="signal", histtype="step")
    mt_svj_hist[n_svj].legend(loc='upper right')


output_file.Close()

# plt.show()


