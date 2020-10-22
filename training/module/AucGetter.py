import module.summaryProcessor as summaryProcessor
import module.utils as utils
import module.trainer as trainer

import time
import os

import pandas as pd
import numpy as np

class AucGetter(object):
    '''This object basically needs to be able to load a training run into memory, including all
    training/testing fractions and random seeds. It then should take a library of signals as input
    and be able to evaluate the auc on each signal to determine a 'general auc' for all signals.
    '''
    
    def __init__(self, filename, summary_path, qcd_path=None, times=False):
        self.times = times
        self.start()
        self.name = summaryProcessor.summary_by_name(summary_path+filename)
        self.d = summaryProcessor.load_summary(self.name)
        self.norm_args = {}
        
        if 'norm_type' in self.d:
            self.norm_args["norm_type"] = str(self.d["norm_type"])
        
        if 'range' in self.d:
            self.norm_args['rng'] = np.asarray(self.d['range'])
        
        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError
        
        self.qcd_path = qcd_path
        
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))
        
        # get and set random seed for reproductions
        self.seed = self.d['seed']
        
        # manually set a bunch of parameters from the summary dict
        for param in ['target_dim', 'input_dim', 'test_split', 'val_split', 'training_output_path']:
            setattr(self, param, self.d[param])
        
        if not os.path.exists(self.training_output_path + ".pkl"):
            print((self.training_output_path + ".pkl"))
            self.training_output_path = utils.path_in_repo(self.training_output_path + ".pkl")
            print((self.training_output_path))
            if self.training_output_path is None:
                raise AttributeError
            else:
                if self.training_output_path.endswith(".h5"):
                    self.training_output_path.rstrip(".h5")
        
        self.instance = trainer.trainer(self.training_output_path)
        self.time('init')
    
    def start(self):
        self.__TIME = time.time()
    
    def time(self, info=None):
        end = time.time() - self.__TIME
        if self.times:
            print((':: TIME: ' + '{}executed in {:.2f} s'.format('' if info is None else info + ' ', end)))
        return end
    
    def update_event_range(self, data, percentile_n, test_key='qcd'):
        utils.set_random_seed(self.seed)
        qcd = getattr(data, test_key).data
        train, test = qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=2)
        rng = utils.percentile_normalization_ranges(train, percentile_n)
        self.norm_args['rng'] = rng
        return rng
    
    def get_test_dataset(self, data, test_key='qcd'):
        self.start()
        assert hasattr(data, test_key), 'please pass a data_holder object instance with attribute \'{}\''.format(
            test_key)
        
        utils.set_random_seed(self.seed)
        
        qcd = getattr(data, test_key).data
        train, test = qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=2)
        
        self.time('test dataset')
        return test
    
    def get_errs_recon(self, data_holder, test_key='qcd', **kwargs):
        test = self.get_test_dataset(data_holder, test_key)
        self.start()
        
        normed = {}
        
        if 'rng' in self.norm_args:
            for key in data_holder.KEYS:
                if key != test_key:
                    normed[key] = getattr(data_holder, key).data.normalize(**self.norm_args)
            normed[test_key] = test.normalize(**self.norm_args)
        else:
            for d, elt in list(data_holder.KEYS.items()):
                if d != test_key:
                    normed[d] = test.normalize(elt.data, **self.norm_args)
            
            normed[test_key] = test.normalize(test, **self.norm_args)
        
        for key in normed:
            normed[key].name = key
        ae = self.instance.load_model()
        normed = list(normed.values())
        err, recon = utils.get_recon_errors(normed, ae, **kwargs)
        for i in range(len(err)):
            err[i].name = err[i].name.rstrip('error').strip()
        
        if 'rng' in self.norm_args:
            for i in range(len(recon)):
                recon[i] = recon[i].inverse_normalize(out_name=recon[i].name, **self.norm_args)
        else:
            for i in range(len(recon)):
                recon[i] = test.inverse_normalize(recon[i], out_name=recon[i].name, **self.norm_args)
        del ae
        self.time('recon gen')
        
        return [{y.name: y for y in x} for x in [normed, err, recon]]
    
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
    
    def plot_aucs(self, err, qcd_key='qcd', metrics=None):
        self.start()
        derr = [v for elt, v in list(err.items()) if elt == qcd_key]
        serr = [v for elt, v in list(err.items()) if elt != qcd_key]
        
        if metrics is None:
            metrics = ['mae']
        ret = utils.roc_auc_plot(data_errs=derr, signal_errs=serr, metrics=metrics)
        self.time('auc plot')
        return ret
    
    def auc_metric(self, aucs):
        data = [(k, v['mae']['auc']) for k, v in list(aucs.items())]
        fmt = pd.DataFrame(data, columns=['name', 'auc'])

        newList = []
        
        for x in fmt.name:
            massAndR = []
            for y in x.split('_')[1:]:
                variable = y.rstrip('GeV')
                variable = variable.replace("p", ".")
                
                massAndR.append(float(variable))
            
            newList.append(massAndR)
        
        mass, nu = np.asarray(newList).T
        nu /= 100
        
        fmt['mass'] = mass
        fmt['nu'] = nu
        
        return fmt