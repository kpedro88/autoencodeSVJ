import module.utils as utils
import module.trainer as trainer
import module.models as models
import module.summaryProcessor as summaryProcessor

from module.dataHolder import data_holder
from module.aucGetter import auc_getter

import numpy as np
import tensorflow as tf
import os
import datetime
import time
from collections import OrderedDict as odict
import pandas as pd
import glob

class aeTrainer:
   
    
    def __init__(self,
                 qcd_path,
                 target_dim,
                 output_data_path,
                 training_params,
                 test_data_fraction=0.15,
                 validation_data_fraction=0.15,
                 custom_objects={},
                 intermediate_architecture=(30, 30),
                 verbose=1,
                 hlf_to_drop=['Energy', 'Flavor'],
                 norm_percentile=1,
                 ):
        """
        Training function for basic auto-encoder (inputs == outputs).
        Will create and save a summary file for this training run, with relevant
        training details etc.

        Not super flexible, but gives a good idea of how good your standard AE is.
        """
        
        start_timestamp = datetime.datetime.now()
        
        seed = np.random.randint(0, 99999999)
        utils.set_random_seed(seed)
        
        # Load QCD samples
        (qcd, qcd_jets, qcd_event, qcd_flavor) = utils.load_all_data(qcd_path, "qcd background",
                                                                     include_hlf=True, include_eflow=True,
                                                                     hlf_to_drop=hlf_to_drop)
        
        # Determine output filename and EFP base
        (filename, EFP_base) = self.get_filename_and_EFP_base(qcd=qcd, target_dim=target_dim,
                                                              output_data_path=output_data_path)
        print(("training under filename '{}'".format(filename)))
        filepath = os.path.join(output_data_path, "trainingRuns", filename)
        
        # Split input data into training, validaiton and test samples
        train_and_validation_data, test_data = qcd.split_by_event(test_fraction=test_data_fraction, random_state=seed,
                                                                  n_skip=len(qcd_jets))
        train_data, validation_data = train_and_validation_data.train_test_split(test_fraction=validation_data_fraction,
                                                                                 random_state=seed)
        
        # Normalize the input
        norm_type = "Custom"
        data_ranges = utils.percentile_normalization_ranges(train_data, norm_percentile)
        
        train_data.name = "qcd training data"
        validation_data.name = "qcd validation data"
        
        train_data_normalized = train_data.normalize(out_name="qcd train norm", rng=data_ranges)
        validation_data_normalized = validation_data.normalize(out_name="qcd val norm", rng=data_ranges)
        
        # Build the model
        input_dim = len(qcd.columns)
        
        model = self.get_auto_encoder_model(input_dim, intermediate_architecture, target_dim)
        
        if verbose:
            model.summary()
            print("TRAINING WITH PARAMS >>>")
            for arg in training_params:
                print((arg, ":", training_params[arg]))
        
        # Run the training
        instance = trainer.trainer(filepath, verbose=verbose)
        
        print("Training the model")
        print("Number of training samples: ", len(train_data_normalized.data))
        print("Number of validation samples: ", len(validation_data_normalized.data))
        
        instance.train(
            x_train=train_data_normalized.data,
            x_test=validation_data_normalized.data,
            y_train=train_data_normalized.data,
            y_test=validation_data_normalized.data,
            model=model,
            force=True,
            use_callbacks=True,
            custom_objects=custom_objects,
            verbose=int(verbose),
            **training_params
        )
        
        end_timestamp = datetime.datetime.now()
        
        # Save training summary
        summary_dict = {
            'target_dim': target_dim,
            'input_dim': input_dim,
            'test_split': test_data_fraction,
            'val_split': validation_data_fraction,
            'hlf': True,
            'eflow': True,
            'eflow_base': EFP_base,
            'seed': seed,
            'filename': filename,
            'filepath': filepath,
            'qcd_path': qcd_path,
            'arch': self.get_architecture_summary(input_dim, intermediate_architecture, target_dim),
            'hlf_to_drop': tuple(hlf_to_drop),
            'start_time': str(start_timestamp),
            'end_time': str(end_timestamp),
            'norm_percentile': norm_percentile,
            'range': data_ranges.tolist(),
            'norm_type': norm_type
        }
        
        summaryProcessor.dump_summary_json(training_params, summary_dict, output_path=(output_data_path + "/summary"))
        
        print("Training executed in: ", (end_timestamp - start_timestamp), " s")

    def get_EFP_base(self, data):
        return len([x for x in data.columns if "eflow" in x])

    def get_filename_and_EFP_base(self, qcd, target_dim, output_data_path):
        """
        Returns a tuple containing filename for given QCD sample (already with correct next version)
        and the EFP base deduced from this QCD sample
        """
        qcd_eflow = self.get_EFP_base(qcd)
    
        eflow_base_lookup = {12: 3, 13: 3, 35: 4, 36: 4}
        eflow_base = eflow_base_lookup[qcd_eflow]
    
        filename = "hlf_eflow{}_{}_".format(eflow_base, target_dim)
    
        last_version = summaryProcessor.get_last_summary_file_version(output_data_path, filename)
        filename += "v{}".format(last_version + 1)
    
        return filename, eflow_base

    def get_auto_encoder_model(self, input_dim, intermediete_architecture, target_dim):
        aes = models.base_autoencoder()
        aes.add(input_dim)
        for elt in intermediete_architecture:
            aes.add(elt, activation='relu')
        aes.add(target_dim, activation='relu')
        for elt in reversed(intermediete_architecture):
            aes.add(elt, activation='relu')
        aes.add(input_dim, activation='linear')
    
        return aes.build()

    def get_architecture_summary(self, input_dim, intermediete_architecture, target_dim):
        arch = (input_dim,) + intermediete_architecture
        arch += (target_dim,)
        arch += tuple(reversed(intermediete_architecture)) + (input_dim,)
        return arch
        