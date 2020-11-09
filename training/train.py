from module.AutoEncoderTrainer import AutoEncoderTrainer

# ------------------------------------------------------------------------------------------------
# This script will run "n_models" auto-encoder trainings with provided hyper-parameters, using
# background data specified in "qcd_path", storing the models in "results_path" and writing
# summaries of trainings to "summary_path".
# ------------------------------------------------------------------------------------------------

output_path = "trainingResults/"
summary_path = output_path+"summary/maxAbsScaler/"
results_path = output_path+"trainingRuns/maxAbsScaler/"

qcd_path = "../../data/training_data/qcd/base_3/*.h5"


# ---------------------------------------------
# Training parameters

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
n_models = 100


# ---------------------------------------------
# Pick normalization type:

# This is a custom implementation of scaling
# norm_type="Custom"
# norm_args = {"norm_percentile" : 25 }


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler
# norm_type="RobustScaler"
# norm_args = {"quantile_range" : (0.25, 0.75),
#              "with_centering" : True,
#              "with_scaling"   : True,
#              "copy"           : True,
#              }

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
# norm_type="MinMaxScaler"
# norm_args = {"feature_range" : (0, 1),
#              "copy"             : True,
#              }

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# norm_type="StandardScaler"
# norm_args = {"with_mean"    : True,
#              "copy"         : True,
#              "with_std"     : True,
#              }

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html#sklearn.preprocessing.MaxAbsScaler
norm_type="MaxAbsScaler"
norm_args = {"copy"         : True,
             }

# ---------------------------------------------
# Run the training

for i in range(n_models):
        
    trainer = AutoEncoderTrainer(qcd_path=qcd_path,
                                 bottleneck_size=target_dim,
                                 training_params=training_params,
                                 norm_type=norm_type,
                                 norm_args=norm_args
                                 )
    
    trainer.run_training(training_output_path=results_path,
                         summaries_path=summary_path,
                         verbose=True
                         )
    
    trainer.save_last_training_summary(path=summary_path)
        
    print('model {} finished'.format(i))