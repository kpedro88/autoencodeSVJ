import module.utils as utils
import module.Trainer as trainer
import module.SummaryProcessor as summaryProcessor
from module.DataProcessor import DataProcessor
from module.DataLoader import DataLoader

import os
import numpy as np
import pandas as pd
from collections import OrderedDict as odict


class AutoEncoderEvaluator:
    
    def __init__(self, summary_path, signals, qcd_path=None):
        """
        Requires path to summaries and signal samples. The QCD path can be deduced from the summary file.
        """
        
        # Set internal variables
        self.d = summaryProcessor.load_summary(summary_path)
        self.set_variables_from_summary()
        self.set_data_paths(qcd_path=qcd_path, signals=signals)
        
        data_loader = DataLoader()
        
        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = data_loader.load_all_data(self.qcd_path, "qcd background",
                                                      include_hlf=self.hlf, include_eflow=self.eflow,
                                                      hlf_to_drop=self.hlf_to_drop
                                                      )
        
        self.find_pkl_file()
        self.trainer = trainer.Trainer(self.training_output_path)
        self.model = self.trainer.load_model()
        
        # Set random seed to the same value as during the training
        utils.set_random_seed(self.seed)
        
        data_processor = DataProcessor(validation_fraction=self.val_split,
                                       test_fraction=self.test_split,
                                       seed=self.seed)
        
        (self.qcd_train_event,
         self.qcd_validation_event,
         self.qcd_test_event,
         train_idx, test_idx) = data_processor.split_to_train_validate_test(data_table=self.qcd_event)
        
        new_train_indices = []
        for index in train_idx:
            new_train_indices.append(2 * index)
            new_train_indices.append(2 * index + 1)
        new_train_indices = pd.Int64Index(new_train_indices)
    
        new_test_indices = []
        for index in test_idx:
            new_test_indices.append(2 * index)
            new_test_indices.append(2 * index + 1)
        new_test_indices = pd.Int64Index(new_test_indices)
        
        (self.qcd_train_data,
         self.qcd_validation_data,
         self.qcd_test_data, _, _) = data_processor.split_to_train_validate_test(data_table=self.qcd,
                                                                                 train_idx=new_train_indices,
                                                                                 test_idx=new_test_indices)
    
        self.qcd_test_data.name = "qcd test data"

        # Calculate means and stds if needed
        
        means_qcd = None
        stds_qcd = None
        
        if self.norm_type == "CustomStandard":
            means_qcd, stds_qcd = self.qcd_test_data.get_means_and_stds()
            

        # Normalize the input
        self.qcd_test_data_normalized = data_processor.normalize(data_table=self.qcd_test_data,
                                                                 normalization_type=self.norm_type,
                                                                 data_ranges=self.norm_ranges,
                                                                 norm_args=self.norm_args,
                                                                 means=means_qcd,
                                                                 stds=stds_qcd)

        means_signal = {}
        stds_signal = {}

        for signal in self.signals:
    
            if self.norm_type == "CustomStandard":
                means_signal[signal], stds_signal[signal] = getattr(self, signal).get_means_and_stds()
            else:
                means_signal[signal], stds_signal[signal] = None, None
            
            setattr(self, signal + '_norm',
                    data_processor.normalize(data_table=getattr(self, signal),
                                             normalization_type=self.norm_type,
                                             data_ranges=self.norm_ranges,
                                             norm_args=self.norm_args,
                                             means=means_signal[signal],
                                             stds=stds_signal[signal]))

        # Get reconstruction values and errors
        qcd_key = "qcd"
        
        data = {qcd_key : self.qcd_test_data_normalized}
        
        for signal in self.signals:
            data[signal] = getattr(self, signal + '_norm')
        
        errors, recons = utils.get_recon_errors(data, self.model)

        self.qcd_err = errors[qcd_key]
        self.qcd_recon = data_processor.normalize(data_table=recons[qcd_key],
                                                  normalization_type=self.norm_type,
                                                  data_ranges=self.norm_ranges,
                                                  norm_args=self.norm_args,
                                                  inverse=True,
                                                  means=means_qcd,
                                                  stds=stds_qcd)

        for signal in self.signals:
            means = None
            stds = None
            
            if signal in means_signal:
                means = means_signal[signal]
                stds = stds_signal[signal]
            
            setattr(self, signal + '_err', errors[signal])
            setattr(self, signal + '_recon',
                    data_processor.normalize(data_table=recons[signal],
                                             normalization_type=self.norm_type,
                                             data_ranges=self.norm_ranges,
                                             norm_args=self.norm_args,
                                             inverse=True,
                                             means=means,
                                             stds=stds))
        
        self.qcd_reps = utils.DataTable(self.model.layers[1].predict(self.qcd_test_data_normalized.data), name='QCD reps')
        
        for signal in self.signals:
            setattr(self, signal + '_reps',
                    utils.DataTable(self.model.layers[1].predict(getattr(self, signal + '_norm').data),
                                    name=signal + ' reps'))
        
        self.qcd_err_jets = [
            utils.DataTable(self.qcd_err.loc[self.qcd_err.index % 2 == i], name=self.qcd_err.name + " jet " + str(i))
            for i in range(2)]
        
        for signal in self.signals:
            serr = getattr(self, signal + '_err')
            setattr(self, signal + '_err_jets',
                    [utils.DataTable(serr.loc[serr.index % 2 == i], name=serr.name + " jet " + str(i)) for i in
                     range(2)])
        
        self.test_flavor = self.qcd_flavor.iloc[self.qcd_test_data.index]
        
        names = list(self.signals.keys())
        
        self.norms_dict = odict([(name, getattr(self, name + '_norm')) for name in names])
        self.norms_dict['qcd'] = self.qcd_test_data_normalized
        
        self.errs_dict = odict([(name, getattr(self, name + '_err')) for name in names])
        self.errs_dict['qcd'] = self.qcd_err
        
        self.all_names = list(self.norms_dict.keys())

    def set_data_paths(self, qcd_path, signals):
        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError
    
        self.qcd_path = qcd_path
    
        assert isinstance(signals, (dict, odict)), 'aux_signals_dict must be dict or odict with {name: path} format'
    
        self.signals = signals
    
        data_loader = DataLoader()
    
        for signal in self.signals:
            setattr(self, signal + '_path', self.signals[signal])
        
            (data,
             jets,
             event,
             flavor) = data_loader.load_all_data(getattr(self, signal + '_path'), signal,
                                                 include_eflow=self.eflow, hlf_to_drop=self.hlf_to_drop,
                                                 include_hlf=self.hlf
                                                 )
            setattr(self, signal, data)
            setattr(self, signal + '_jets', jets)
            setattr(self, signal + '_event', event)
            setattr(self, signal + '_flavor', flavor)

    def set_variables_from_summary(self):
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))
        self.seed = self.d['seed']
        self.target_dim = self.d['target_dim']
        self.input_dim = self.d['input_dim']
        self.test_split = self.d['test_split']
        self.val_split = self.d['val_split']
        self.training_output_path = self.d['training_output_path']
        self.norm_type = self.d["norm_type"]
        self.norm_ranges = np.asarray(self.d["range"])
        self.norm_args = self.d['norm_args']
        
        if 'norm_means_test' in self.d and 'norm_stds_test' in self.d:
            self.means_test = self.d['norm_means_test']
            self.stds_test = self.d['norm_stds_test']
        else:
            self.means_test = None
            self.stds_test = None
    
        print("norm type:", self.norm_type)
        print("norm args: ", self.norm_args)

    def find_pkl_file(self):
        # try to find training pkl file and load 'er up
        if not os.path.exists(self.training_output_path + ".pkl"):
            print((self.training_output_path + ".pkl"))
            self.filepath = utils.path_in_repo(self.training_output_path + ".pkl")
            print((self.training_output_path))
            if self.training_output_path is None:
                raise AttributeError
            else:
                if self.training_output_path.endswith(".h5"):
                    self.training_output_path.rstrip(".h5")
    
    def roc(self, show_plot=True, metrics=None, figsize=8, figloc=(0.3, 0.2), *args, **kwargs):
        
        if metrics is None:
            metrics=['mae', 'mse']
        
        qcd = self.errs_dict['qcd']
        others = [self.errs_dict[n] for n in self.all_names if n != 'qcd']
   
        if show_plot:
            utils.roc_auc_plot(qcd, others, metrics=metrics, figsize=figsize, figloc=figloc, *args, **kwargs)
            return
        
        return utils.roc_auc_dict(qcd, others, metrics=metrics)
