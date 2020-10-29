from module.AutoEncoderTrainer import AutoEncoderTrainer

# ------------------------------------------------------------------------------------------------
# This script will run "n_models" auto-encoder trainings with provided hyper-parameters, using
# background data specified in "qcd_path", storing the models in "results_path" and writing
# summaries of trainings to "summary_path".
# ------------------------------------------------------------------------------------------------

output_path = "trainingResults/"
summary_path = output_path+"summary/test/"
results_path = output_path+"trainingRuns/test/"

qcd_path = "../../data/training_data/qcd/base_3/*.h5"

training_params = {
    'batch_size': 32,
    'loss': 'mse',
    'optimizer': 'adam',
    'epochs': 2,
    'learning_rate': 0.00051,
    'es_patience': 12,
    'lr_patience': 9,
    'lr_factor': 0.5
}

target_dim = 8
norm_percentile = 25
n_models = 1

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