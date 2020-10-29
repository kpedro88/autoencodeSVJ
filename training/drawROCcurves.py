import module.SummaryProcessor as summaryProcessor
from module.AutoEncoderEvaluator import AutoEncoderEvaluator

# ------------------------------------------------------------------------------------------------
# This script will draw ROC curves for a specified model against all signals found in the
# "signals_base_path". If no version is specified (set to None), the latest training
# will be used.
# ------------------------------------------------------------------------------------------------

training_version = None
efp_base = 3
bottleneck_dim = 8
summaries_path = "trainingResults/summary/"
summary_base_name = "hlf_eflow{}_{}_".format(efp_base, bottleneck_dim)

input_summary_path = summaryProcessor.get_latest_summary_file_path(summaries_path=summaries_path,
                                                                   file_name_base=summary_base_name,
                                                                   version=training_version)


# masses = [1500, 2000, 2500, 3000, 3500, 4000]
masses = [2000]
rinvs = [0.15, 0.30, 0.45, 0.60, 0.75]

signals_base_path = "../../data/training_data/all_signals/"

signals = {"{}, {}".format(mass, rinv) : "{}{}GeV_{:1.2f}/base_3/*.h5".format(signals_base_path, mass, rinv)
           for mass in masses
           for rinv in rinvs}

evaluator = AutoEncoderEvaluator(input_summary_path, signals=signals)
evaluator.roc(xscale='log', metrics=["mae"])