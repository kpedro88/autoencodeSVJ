import module.SummaryProcessor as summaryProcessor

# ------------------------------------------------------------------------------------------------
# This script will produce a CSV file with areas under ROC curves (AUCs) for each training
# summary file found in the "summary_path" below, testing on all signal samples found
# in "input_path". The output will be stored in "aucs_path".
# ------------------------------------------------------------------------------------------------

summary_path = "trainingResults/summary/"
input_path = "../../data/s_channel_delphes/h5_signal_no_MET_over_mt_cut/*.h5"

AUCs_path = "trainingResults/aucs/"

summaryProcessor.save_all_missing_AUCs(summary_path=summary_path, signals_path=input_path, AUCs_path=AUCs_path)
