import module.utils as utils
import module.Trainer as trainer
import module.SummaryProcessor as summaryProcessor

import os
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
        
        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(self.qcd_path, "qcd background",
                                                include_hlf=self.hlf, include_eflow=self.eflow,
                                                hlf_to_drop=self.hlf_to_drop
                                                )
        
        self.find_pkl_file()
        self.trainer = trainer.Trainer(self.training_output_path)
        self.model = self.trainer.load_model()
        
        # Set random seed to the same value as during the training
        utils.set_random_seed(self.seed)
        
        # Split input data into training, validaiton and test samples
        self.qcd_train_and_validation_data, self.qcd_test_data = self.qcd.split_by_event(test_fraction=self.test_split,
                                                                                         random_state=self.seed,
                                                                                         n_skip=len(self.qcd_jets))

        self.qcd_train_data, self.qcd_validation_data = self.qcd_train_and_validation_data.train_test_split(self.val_split,
                                                                                                            self.seed)
        
        # Normalize the input
        if self.norm_type == "Custom":
            self.data_ranges = utils.percentile_normalization_ranges(self.qcd_test_data, self.norm_percentile)
            
            self.qcd_train_data.name = "qcd training data"
            self.qcd_test_data.name = "qcd test data"
            self.qcd_validation_data.name = "qcd validation data"
            
            self.qcd_train_data_normalized = self.qcd_train_data.normalize_in_range(rng=self.data_ranges)
            self.qcd_validation_data_normalized = self.qcd_validation_data.normalize(rng=self.data_ranges)
            self.qcd_test_data_normalized = self.qcd_test_data.normalize(rng=self.data_ranges)
            
            for signal in self.signals:
                setattr(self, signal + '_norm',
                        getattr(self, signal).normalize(out_name=signal + ' norm', rng=self.data_ranges))
        else:
            print("Normalization not implemented: ", self.norm_type)
        
        # self.train_data_normalized = self.train_data.norm(out_name="qcd train norm", **self.norm_args)
        
        # Get reconstruction values and errors
        data = [self.qcd_test_data_normalized]
        
        for signal in self.signals:
            data.append(getattr(self, signal + '_norm'))
        
        errors, recons = utils.get_recon_errors(data, self.model)
        self.qcd_err, signal_errs = errors[0], errors[1:]
        
        if self.norm_type == "Custom":
            self.qcd_recon = recons[0].inverse_normalize_in_range(rng=self.data_ranges)
            
            for err, recon, signal in zip(signal_errs, recons[1:], self.signals):
                setattr(self, signal + '_err', err)
                setattr(self, signal + '_recon', recon.inverse_normalize_in_range(rng=self.data_ranges))
        else:
            print("Normalization not implemented: ", self.norm_type)
        
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
    
        for signal in self.signals:
            setattr(self, signal + '_path', self.signals[signal])
        
            (data,
             jets,
             event,
             flavor) = utils.load_all_data(getattr(self, signal + '_path'), signal,
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
        self.norm_percentile = self.d["norm_percentile"]
    
        self.norm_args = {"norm_type": self.norm_type, "norm_percentile": self.norm_percentile}

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
