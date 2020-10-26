import module.utils as utils
from module.logger import logger

from enum import Enum
from sklearn.model_selection import train_test_split

import sklearn.preprocessing as prep
import pandas as pd
import numpy as np
import chardet
import glob

class DataTable(logger):
    """
        wrapper for the pandas data table. Allows for quick variable plotting and train/test/splitting.
    """
    
    class NormTypes(Enum):
        MinMaxScaler = 0
        StandardScaler = 1
        RobustScaler = 2
    
    norm_types_dict = dict([(x.value, x.name) for x in NormTypes])
    table_count = 0
    
    def __init__(self, data, headers=None, name=None, verbose=True):
        
        logger.__init__(self, "data_table :: ", verbose)
        self.name = name or "untitled {}".format(DataTable.table_count)
        DataTable.table_count += 1
        
        if headers is not None:
            self.headers = headers
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.headers = data.columns
            self.data = data
        elif isinstance(data, DataTable):
            self.headers = data.headers
            self.data = data.df.values
            self.name = data.name
        else:
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)
            
            self.headers = ["dist " + str(i + 1) for i in range(data.shape[1])]
            self.data = data
        
        assert len(self.data.shape) == 2, "data must be matrix!"
        assert len(self.headers) == self.data.shape[1], "n columns must be equal to n column headers"
        assert len(self.data) > 0, "n samples must be greater than zero"
        self.scaler = None
        if isinstance(self.data, pd.DataFrame):
            self.df = self.data
            self.data = self.df.values
        else:
            self.df = pd.DataFrame(self.data, columns=self.headers)
    
    def normalize(self, data=None, norm_type=0, out_name=None, rng=None, **scaler_args):
        
        if rng is not None:
            return self.normalize_in_range(rng, out_name)
        
        if isinstance(norm_type, str):
            norm_type = getattr(self.NormTypes, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NormTypes, self.norm_types_dict[norm_type])
        
        assert isinstance(norm_type, self.NormTypes)
        
        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, DataTable), "data must be data_table type"
        
        if out_name is None:
            out_name = "'{}' normed to '{}'".format(data.name, self.name)
        
        ret = DataTable(pd.DataFrame(self.scaler.transform(data.df), columns=data.df.columns, index=data.df.index),
                        name=out_name)
        return ret
    
    def inverse_normalize(self, data=None, norm_type=0, out_name=None, rng=None, **scaler_args):
        if rng is not None:
            return self.inverse_normalize_in_range(rng, out_name)
        if isinstance(norm_type, str):
            norm_type = getattr(self.NormTypes, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NormTypes, self.norm_types_dict[norm_type])
        
        assert isinstance(norm_type, self.NormTypes)
        
        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, DataTable), "data must be data_table type"
        
        if out_name is None:
            out_name = "'{}' inv_normed to '{}'".format(data.name, self.name)
        
        ret = DataTable(
            pd.DataFrame(self.scaler.inverse_transform(data.df), columns=data.df.columns, index=data.df.index),
            name=out_name)
        
        # ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
        return ret
    
    def normalize_in_range(self, rng, out_name=None):
        if out_name is None:
            out_name = "{} norm".format(self.name)
        
        return DataTable((self.df - rng[:, 0]) / (rng[:, 1] - rng[:, 0]), name=out_name)
    
    def inverse_normalize_in_range(self, rng, out_name=None):
        if out_name is None:
            if self.name.endswith('norm'):
                out_name = self.name.replace('norm', '').strip()
            else:
                out_name = "{} inverse normed".format(self.name)

        return DataTable(self.df * (rng[:, 1] - rng[:, 0]) + rng[:, 0], name=out_name)
    
    def __getattr__(self, attr):
        if hasattr(self.df, attr):
            return self.df.__getattr__(attr)
        else:
            raise AttributeError
    
    def __getitem__(self, item):
        return self.df[item]
    
    def __str__(self):
        return self.df.__str__()
    
    def __repr__(self):
        return self.df.__repr__()
    
    def split_by_column_names(self, column_list_or_criteria):
        match_list = None
        if isinstance(column_list_or_criteria, str):
            match_list = [c for c in self.headers if glob.fnmatch.fnmatch(c, column_list_or_criteria)]
        else:
            match_list = list(column_list_or_criteria)
        
        other = [c for c in self.headers if c not in match_list]
        
        t1, t2 = self.df.drop(other, axis=1), self.df.drop(match_list, axis=1)
        
        return DataTable(t1, headers=match_list, name=self.name), DataTable(t2, headers=other, name=self.name)

    def train_test_split(self, test_fraction=0.25, random_state=None):
        dtrain, dtest = train_test_split(self, test_size=test_fraction, random_state=random_state)
        return DataTable(dtrain, name="train"), DataTable(dtest, name="test")
    
    def split_by_event(self, test_fraction=0.25, random_state=None, n_skip=2):
        # shuffle event indicies
        train_idx, test_idx = train_test_split(self.df.index[0::n_skip], test_size=test_fraction,
                                               random_state=random_state)
        train, test = [np.asarray([x + i for i in range(n_skip)]).T.flatten() for x in [train_idx, test_idx]]
        return DataTable(self.df.loc[train], name="train"), DataTable(self.df.loc[test], name="test")

    def cdrop(self, globstr, inplace=False):
        to_drop = list(utils.parse_globlist(globstr, list(self.df.columns)))
        
        if inplace:
            modify = self
        else:
            ret = DataTable(self)
            modify = ret

        first_axis_label = modify.df.axes[1][0]

        for i, d in enumerate(to_drop):
            if type(d) is str and type(first_axis_label) is bytes:
                axis_encoding = chardet.detect(first_axis_label)["encoding"]
                to_drop[i] = d.encode(axis_encoding)
            elif type(d) is np.bytes_ and type(first_axis_label) is bytes:
                dd = d.decode(chardet.detect(d)["encoding"])
                axis_encoding = chardet.detect(first_axis_label)["encoding"]
                ddd = dd.encode(axis_encoding)
                to_drop[i] = ddd
            elif type(d) is np.bytes_ and type(first_axis_label) is str:
                encoding = chardet.detect(d)["encoding"]
                to_drop[i] = d.decode(encoding)
            else:
                to_drop[i] = d

        modify.df.drop(to_drop, axis=1, inplace=True)
        
        modify.headers = list(modify.df.columns)
        modify.data = np.asarray(modify.df)
        return modify
    
    def cfilter(self, globstr, inplace=False):
        to_keep = utils.parse_globlist(globstr, list(self.df.columns))
        to_drop = set(self.headers).difference(to_keep)
        
        to_drop = list(to_drop)
        
        modify = None
        if inplace:
            modify = self
        else:
            ret = DataTable(self)
            modify = ret
        
        dummy = []
        
        for col in modify.df.axes[1]:
            if type(col) is bytes:
                encoding = chardet.detect(col)["encoding"]
                dummy.append(col.decode(encoding))
            else:
                dummy.append(col)
        
        modify.df.set_axis(dummy, axis=1, inplace=True)
        
        first_axis_label = modify.df.axes[1][0]
        
        for i, d in enumerate(to_drop):
            if type(d) is str and type(first_axis_label) is bytes:
                axis_encoding = chardet.detect(first_axis_label)["encoding"]
                to_drop[i] = d.encode(axis_encoding)
            elif type(d) is np.bytes_ and type(first_axis_label) is bytes:
                dd = d.decode(chardet.detect(d)["encoding"])
                axis_encoding = chardet.detect(first_axis_label)["encoding"]
                ddd = dd.encode(axis_encoding)
                to_drop[i] = ddd
            elif type(d) is np.bytes_ and type(first_axis_label) is str:
                encoding = chardet.detect(d)["encoding"]
                to_drop[i] = d.decode(encoding)
            else:
                to_drop[i] = d
        
        for d in to_drop:
            modify.df.drop(d, axis=1, inplace=True)
        
        modify.headers = list(modify.df.columns)
        return modify
    
    def cmerge(self, other, out_name):
        assert self.shape[0] == other.shape[0], 'data tables must have same number of samples'
        return DataTable(self.df.join(other.df), name=out_name)