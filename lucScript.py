import autoencode.module.autoencodeSVJ.evaluate as ev
import autoencode.module.autoencodeSVJ.summaryProcessor as summaryProcessor

import time
import datetime

import pandas as pd

TRAIN = True

lr = .00051
lr_factor = 0.5
es_patience = 12
target_dim = 8
batch_size = 32
norm_percentile = 25
epochs = 100
n_models = 50

output_path = "trainingResults/"
summary_path = output_path+"summary/"
results_path = output_path+"trainingRuns/"

input_path = "../data/training_data/"
signal_path = input_path + "2000GeV_0.75/base_3/*.h5"
qcd_path = input_path + "qcd/base_3/*.h5"

start_stamp = time.time()
res = None
if TRAIN:
    for i in range(n_models):
        mse = ev.ae_train(
            output_data_path=output_path,
            signal_path=signal_path,
            qcd_path=qcd_path,
            target_dim=target_dim,
            verbose=True,
            batch_size=batch_size,
            learning_rate=lr,
            norm_percentile=norm_percentile,
            lr_factor=lr_factor,
            es_patience=es_patience,
            epochs=epochs
        )
    
    res = summaryProcessor.summary(summary_path=summary_path)
    res = res[pd.DatetimeIndex(res.start_time) > datetime.datetime.fromtimestamp(start_stamp)]

else:
    res = summaryProcessor.summary(summary_path=summary_path)
    res = res.sort_values('start_time').tail(n_models)

# take lowest 10% losses of all trainings
n_best = int(0.1 * n_models)
best_ = res.sort_values('total_loss').head(n_best)
best_name = str(best_.summary_path.values[0])

print("best trainings: ", best_)
print("best training name: ", best_name)