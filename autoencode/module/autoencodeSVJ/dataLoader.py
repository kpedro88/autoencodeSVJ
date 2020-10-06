import autoencode.module.autoencodeSVJ.utils as utils

from autoencode.module.autoencodeSVJ.logger import logger
from autoencode.module.autoencodeSVJ.dataTable import data_table

from collections import OrderedDict as odict
from operator import mul
from functools import reduce

import os
import h5py
import glob
import numpy as np

class data_loader(logger):
    """
    data loader / handler/merger for h5 files with the general format
    of this repository
    """
    
    def __init__(self, name, verbose=True):
        logger.__init__(self)
        self.name = name
        self._LOG_PREFIX = "data_loader :: "
        self.VERBOSE = verbose
        self.samples = odict()
        self.sample_keys = None
        self.data = odict()
        self.labels = odict()
    
    def add_sample(self, sample_path):
        filepath = utils.smartpath(sample_path)
        
        assert os.path.exists(filepath)
        
        if filepath not in self.samples:
            with h5py.File(filepath, mode="r") as f:
                
                self.log("Adding sample at path '{}'".format(filepath))
                self.samples[filepath] = f
                
                keys = set(f.keys())
                
                if self.sample_keys is None:
                    self.sample_keys = keys
                else:
                    if keys != self.sample_keys:
                        raise AttributeError
                
                self._update_data(f, keys)
    
    def make_table(self, key, name=None, third_dim_handle="stack"):
        """ stack, combine, or split """
        assert third_dim_handle in ['stack', 'combine', 'split']
        assert key in self.sample_keys
        
        data = self.data[key]
        labels = self.labels[key]
        name = name or self.name
        
        if len(data.shape) == 1:
            return data_table(np.expand_dims(data, 1), headers=labels, name=name)
        elif len(data.shape) == 2:
            return data_table(data, headers=labels, name=name)
        elif len(data.shape) == 3:
            ret = data_table(
                np.vstack(data),
                headers=labels,
                name=name
            )
            # isa jet behavior
            if third_dim_handle == 'stack':
                # stack behavior
                return ret
            elif third_dim_handle == 'split':
                if key.startswith("jet"):
                    prefix = "jet"
                else:
                    prefix = "var"
                
                return [
                    data_table(
                        ret.iloc[i::data.shape[1]],
                        name="{} {} {}".format(ret.name, prefix, i)
                    ) for i in range(data.shape[1])
                ]
                
                # [
                #     data_table(
                #         data[:,i,:],
                #         headers=labels,
                #         name="{}_{}".format(name,i)
                #     ) for i in range(data.shape[1])
                # ]
            else:
                prefix = 'jet' if key.startswith('jet') else 'var'
                return data_table(
                    self.stack_data(data, axis=1),
                    headers=self.stack_labels(labels, data.shape[1], prefix),
                    name=name,
                )
                # combine behavior
        else:
            raise AttributeError
    
    def make_tables(
            self,
            keylist,
            name,
            third_dim_handle="stack",
    ):
        tables = []
        for k in keylist:
            tables.append(self.make_table(k, None, third_dim_handle))
        assert len(tables) > 0
        ret, tables = tables[0], tables[1:]
        for table in tables:
            if third_dim_handle == "split":
                for i, (r, t) in enumerate(zip(ret, table)):
                    ret[i] = r.cmerge(t, name + str(i))
            else:
                ret = ret.cmerge(table, name)
        return ret
    
    def stack_data(
            self,
            data,
            axis=1,
    ):
        return np.hstack(np.asarray(np.split(data, data.shape[axis], axis=axis)).squeeze())
    
    def stack_labels(
            self,
            labels,
            n,
            prefix,
    ):
        new = []
        for j in range(n):
            for l in labels:
                new.append("{}{}_{}".format(prefix, j, l))
        return np.asarray(new)
    
    def get_dataset(
            self,
            keys=None,
    ):
        if keys is None:
            raise AttributeError
        
        assert hasattr(keys, "__iter__") or isinstance(keys, str), "keys must be iterable of strings!"
        
        if isinstance(keys, str):
            keys = [keys]
        
        data_dict = self.data.copy()
        shapes = []
        for key in data_dict:
            keep = False
            for subkey in keys:
                if glob.fnmatch.fnmatch(key, subkey):
                    keep = True
            if not keep:
                del data_dict[key]
            else:
                shapes.append(data_dict[key].shape)
        
        types = [v.dtype.kind for v in list(data_dict.values())]
        is_string = [t == 'S' for t in types]
        if any(is_string):
            if all(is_string):
                return np.concatenate(list(data_dict.values())), data_dict
            raise AttributeError
        
        self.log("Grabbing dataset with keys {0}".format(list(data_dict.keys())))
        
        samples = set([x.shape[0] for x in list(data_dict.values())])
        assert len(samples) == 1, "all datasets with matching keys need to have IDENTICAL sizes!"
        sample_size = samples.pop()
        
        sizes = [reduce(mul, x.shape[1:], 1) for x in list(data_dict.values())]
        splits = [0, ] + [sum(sizes[:i + 1]) for i in range(len(sizes))]
        
        dataset = np.empty((sample_size, sum(sizes)))
        
        for i, datum in enumerate(data_dict.values()):
            dataset[:, splits[i]:splits[i + 1]] = datum.reshape(datum.shape[0], sizes[i])
        # self.log("Dataset shape: {0}".format(dataset.shape))
        return dataset, data_dict
    
    def save(
            self,
            filepath,
            force=False,
    ):
        """saves the current sample sets as one h5 file to the filepath specified"""
        filepath = utils.smartpath(filepath)
        if not filepath.endswith('.h5'):
            filepath += '.h5'
        
        if os.path.exists(filepath) and not force:
            self.error("Path '{0}' already contains data. Use the 'force' argument to force saving".format(filepath))
            return 1
        
        f = h5py.File(filepath, "w")
        for key, data in list(self.data.items()):
            f.create_dataset(key, data=data)
        
        self.log("Saving current dataset to file '{0}'".format(filepath))
        f.close()
        return 0
    
    def _update_data(
            self,
            sample_file,
            keys_to_add,
    ):
        for key in keys_to_add:
            assert 'data' in sample_file[key]
            assert 'labels' in sample_file[key]
            
            if key not in self.labels:
                self.labels[key] = np.asarray(sample_file[key]['labels'])
            else:
                assert (self.labels[key] == np.asarray(sample_file[key]['labels'])).all()
            
            if key not in self.data:
                self.data[key] = np.asarray(sample_file[key]['data'])
            else:
                self.data[key] = np.concatenate([self.data[key], sample_file[key]['data']])