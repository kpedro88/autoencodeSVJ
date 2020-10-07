import pandas as pd
import json 
import numpy as np
from collections import OrderedDict as odict
from scipy.stats import ks_2samp 

class dtrans:
    def __init__(
        self,
        dict_in,
    ):
        self.data = None
        self.loss = None
        self.val_loss = None
        self.pred = None
        self.true = None
        self.params = None

        if isinstance(dict_in, str):
            with open(dict_in, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = dict_in
        
        self.keys = list(self.data.keys())
        self.key_indicies = odict()
        for i,key in enumerate(self.keys):
            self.key_indicies[key] = i
        
        for attr in ['loss', 'val_loss']:
            setattr(self, attr, np.asarray([self.data[key]['out']['history'][attr] for key in self.keys]))

        for attr in ['pred', 'true']:
            setattr(self, attr, np.asarray([self.data[key]['out'][attr] for key in self.keys]))

        self.params = pd.DataFrame(
            [list(self.data[key]['params'].values()) for key in self.keys],
            columns=list(self.data[self.keys[0]]['params'].keys())
        )
        self.params['loss'] = [elt.replace('error', 'err').replace('mean_', '').replace('absolute', 'abs').replace('percentage', 'pct').replace('squared', 'sq').replace('logarithmic', 'log')  for elt in self.params.loss]
        self.params['train_loss'] = self.loss.min(axis=1).tolist()
        self.params['val_loss'] = self.val_loss.min(axis=1).tolist()
        self.params['combined_loss'] = self.params.train_loss + self.params.val_loss
        self.true = self.true[0]

        ret = np.asarray([np.mean([ks_2samp(self.pred[j,:,i], self.true[:,i]) for i in range(self.true.shape[1])], axis=0) for j in range(self.pred.shape[0])])

        self.params['hist_stat'] = ret[:,0]
        self.params['hist_chi2'] = ret[:,1]

        self.index = np.argsort(self.params.hist_chi2)[::-1]
        self.params = self.params.sort_values('hist_chi2')[::-1]
        self.subindex = self.params.combined_loss >= 0
        self.params = self.params[self.subindex]
        self.loss = self.loss[self.index][self.subindex]
        self.val_loss = self.val_loss[self.index][self.subindex]
        self.pred = self.pred[self.index][self.subindex]

        self.params.drop('epochs', axis=1)
