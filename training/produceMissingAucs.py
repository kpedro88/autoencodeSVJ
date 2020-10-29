import module.SummaryProcessor as summaryProcessor

# ------------------------------------------------------------------------------------------------
# This script will produce a CSV file with areas under ROC curves (AUCs) for each training
# summary file found in the "summary_path" below, testing on all signal samples found
# in "input_path". The output will be stored in "aucs_path".
# ------------------------------------------------------------------------------------------------

summary_path = "trainingResults/summary/test/"
input_path = "../../data/training_data/all_signals/*/base_3/*.h5"

AUCs_path = "trainingResults/aucs/test/"

summaryProcessor.save_all_missing_AUCs(summary_path=summary_path, signals_path=input_path, AUCs_path=AUCs_path)
