import module.utils as utils

import collections
import os
import pickle

from pathlib import Path

class pkl_file(collections.MutableMapping):
    """Dictionary which saves all attributes to a .pkl file on access/altering"""
    
    def __init__(self, path, verbose=1, *args, **kwargs):
        
        if not path.endswith(".pkl"):
            path += ".pkl"
        
        self.path = utils.smartpath(path)
        self.store = {}
        
        if os.path.exists(self.path):
            try:
                self.update_store()
            except:
                raise AttributeError("failed to load pickle file!")
        
        self.update_pkl()
    
    def __getitem__(self, key):
        self.update_store()
        return self.store[key]
    
    def __setitem__(self, key, value):
        self.update_store()
        self.store[key] = value
        self.update_pkl()
    
    def __delitem__(self, key):
        self.update_store()
        del self.store[key]
        self.update_pkl()
    
    def __iter__(self):
        self.update_store()
        return iter(self.store)
    
    def __len__(self):
        self.update_store()
        return len(self.store)
    
    def update_pkl(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'wb') as f:
            pickle.dump(self.store, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def update_store(self):
        with open(self.path, 'rb') as f:
            self.store.update(pickle.load(f))
    
    def __str__(self):
        self.update_store()
        return str(self.store)
    
    def __repr__(self):
        self.update_store()
        return "pkl_file instance at {}\n".format(self.path) + str(self)