import module.SummaryProcessor as summaryProcessor
import module.utils as utils
import module.Trainer as trainer
from module.DataProcessor import DataProcessor

import time
import os

import pandas as pd
import numpy as np


class AucGetter(object):
    """This object basically needs to be able to load a training run into memory, including all
    training/testing fractions and random seeds. It then should take a library of signals as input
    and be able to evaluate the auc on each signal to determine a 'general auc' for all signals.
    """
    
    def __init__(self, filename, summary_path, print_times=False):
        
        self.print_times = print_times
        self.start()
        
        self.name = summaryProcessor.summary_by_name(summary_path+filename)
        self.d = summaryProcessor.load_summary(self.name)
        
        self.set_variables_from_summary()
        
        if not os.path.exists(self.training_output_path + ".pkl"):
            print((self.training_output_path + ".pkl"))
            self.training_output_path = utils.path_in_repo(self.training_output_path + ".pkl")
            print(self.training_output_path)
            if self.training_output_path is None:
                raise AttributeError
            else:
                if self.training_output_path.endswith(".h5"):
                    self.training_output_path.rstrip(".h5")
        
        self.trainer = trainer.Trainer(self.training_output_path)
        self.time('init')
    
    def set_variables_from_summary(self):
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))
        self.seed = self.d['seed']
        self.test_split = self.d['test_split']
        self.validation_split = self.d['val_split']
        self.qcd_path = self.d['qcd_path']
        self.target_dim = self.d['target_dim']
        self.input_dim = self.d['input_dim']
        self.training_output_path = self.d['training_output_path']
        self.norm_type = self.d["norm_type"]
        self.norm_ranges = np.asarray(self.d["range"])
        self.norm_args = self.d['norm_args']
    
    def start(self):
        self.__TIME = time.time()
    
    def time(self, info=None):
        end = time.time() - self.__TIME
        if self.print_times:
            print((':: TIME: ' + '{}executed in {:.2f} s'.format('' if info is None else info + ' ', end)))
        return end
        
    def get_test_dataset(self, data_holder, test_key='qcd'):
        self.start()
        assert hasattr(data_holder, test_key), 'please pass a data_holder object instance with attribute \'{}\''.format(
            test_key)
        
        utils.set_random_seed(self.seed)
        
        qcd = getattr(data_holder, test_key).data
        
        data_processor = DataProcessor(validation_fraction=self.validation_split,
                                       test_fraction=self.test_split,
                                       seed=self.seed)
        
        train, validation, test, _, _ = data_processor.split_to_train_validate_test(data_table=qcd)
        
        self.time('test dataset')
        return test
    
    def get_errs_recon(self, data_holder, test_key='qcd', **kwargs):
        
        self.start()
        
        data_processor = DataProcessor(seed=self.seed)
        normed = {}

        
        means = {}
        stds = {}

        test = self.get_test_dataset(data_holder, test_key)
        
        if self.norm_type == "CustomStandard":
            means[test_key], stds[test_key] = test.get_means_and_stds()
        else:
            means[test_key], stds[test_key] = None, None

        
        normed[test_key]= data_processor.normalize(data_table=test,
                                                   normalization_type=self.norm_type,
                                                   data_ranges=self.norm_ranges,
                                                   norm_args=self.norm_args,
                                                   means=means[test_key],
                                                   stds=stds[test_key])


        qcd_scaler = test.scaler

        for key in data_holder.KEYS:
            if key != test_key:
                data = getattr(data_holder, key).data

                if self.norm_type == "CustomStandard":
                    means[key], stds[key] = data.get_means_and_stds()
                else:
                    means[key], stds[key] = None, None
                
                normed[key] = data_processor.normalize(data_table=data,
                                                       normalization_type=self.norm_type,
                                                       data_ranges=self.norm_ranges,
                                                       norm_args=self.norm_args,
                                                       means=means[key],
                                                       stds=stds[key],
                                                       scaler=qcd_scaler)
        
        for key in normed:
            normed[key].name = key
        
        auto_encoder = self.trainer.load_model()
        
        err, recon = utils.get_recon_errors(normed, auto_encoder, **kwargs)
        
        for key, value in err.items():
            err[key].name = value.name.rstrip('error').strip()
        
        for key, value in recon.items():
            recon[key] = data_processor.normalize(data_table=value,
                                                  normalization_type=self.norm_type,data_ranges=self.norm_ranges,
                                                  norm_args=self.norm_args,
                                                  inverse=True,
                                                  means=means[key],
                                                  stds=stds[key],
                                                  scaler=qcd_scaler)

        del auto_encoder
        self.time('recon gen')
        
        return [{z.name: z for y, z in x.items()} for x in [normed, err, recon]]
    
    def get_aucs(self, errors, qcd_key='qcd', metrics=None):
        self.start()

        if metrics is None:
            metrics = ['mae']

        background_errors = []
        signal_errors = []

        for key, value in list(errors.items()):
            if key == qcd_key:
                background_errors.append(value)
            else:
                signal_errors.append(value)

        ROCs_and_AUCs_per_signal = utils.roc_auc_dict(data_errs=background_errors, signal_errs=signal_errors, metrics=metrics)
        
        self.time('auc grab')
        return ROCs_and_AUCs_per_signal
    
    def auc_metric(self, aucs):
        data = [(k, v['mae']['auc']) for k, v in list(aucs.items())]
        fmt = pd.DataFrame(data, columns=['name', 'auc'])

        new_list = []
        
        for x in fmt.name:
            mass_and_r = []
            for y in x.split('_')[1:]:
                variable = y.rstrip('GeV')
                variable = variable.replace("p", ".")
                
                mass_and_r.append(float(variable))
            
            new_list.append(mass_and_r)
        
        mass, nu = np.asarray(new_list).T
        nu /= 100
        
        fmt['mass'] = mass
        fmt['nu'] = nu
        
        return fmt
