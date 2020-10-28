from module.AutoEncoderTrainer import AutoEncoderTrainer
import module.SummaryProcessor as summaryProcessor

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import time
import datetime

output_path = "trainingResults/"
aucs_path = output_path+"aucs/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../../data/training_data/"
signal_path = input_path + "all_signals/2000GeV_0.75/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

TRAIN = True

training_params = {
    'batch_size': 32,
    'loss': 'mse',
    'optimizer': 'adam',
    'epochs': 200,
    'learning_rate': 0.00051,
    'es_patience': 12,
    'lr_patience': 9,
    'lr_factor': 0.5
}


target_dim = 8
norm_percentile = 25

n_models = 200  # number of models to train
model_acceptance_fraction = 10  # take top N best performing models

start_stamp = time.time()
res = None

if TRAIN:
    for i in range(n_models):
        
        trainer = AutoEncoderTrainer(qcd_path=qcd_path,
                                     bottleneck_size=target_dim,
                                     training_params=training_params,
                                     norm_percentile=norm_percentile
                                     )
    
        trainer.run_training(training_output_path=results_path,
                             summaries_path=summary_path,
                             verbose=True
                             )
    
        trainer.save_last_training_summary(path=summary_path)
        
        print('model {} finished'.format(i))

    res = summaryProcessor.summary(summary_path=summary_path)
    res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime.fromtimestamp(start_stamp)]

else:
    res = summaryProcessor.summary(summary_path=summary_path)
    res = res.sort_values('start_time').tail(n_models)

print("res: ", res)

# take lowest 10% losses of all trainings
# n_best = int(0.01 * model_acceptance_fraction * n_models)
# best_ = res.sort_values('total_loss').head(n_best)
# best_name = str(best_.filename.values[0])
#
# print("N best: ", n_best)
# print("Best models: ", best_)
# print("The best model: ", best_name)

summaryProcessor.save_all_missing_AUCs(summary_path=summary_path,
                                       signals_path=(input_path + "all_signals/*/base_3/*.h5"),
                                       AUCs_path=aucs_path)