import module.utils as utils
from module.logger import logger

from enum import Enum
from sklearn.model_selection import train_test_split

import sklearn.preprocessing as prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet
import glob

class data_table(logger):
    class NORM_TYPES(Enum):
        MinMaxScaler = 0
        StandardScaler = 1
        RobustScaler = 2
    
    _RDICT_NORM_TYPES = dict([(x.value, x.name) for x in NORM_TYPES])
    
    TABLE_COUNT = 0
    
    """
    wrapper for the pandas data table.
    allows for quick variable plotting and train/test/splitting.
    """
    
    def __init__(
            self,
            data,
            headers=None,
            name=None,
            verbose=1,
    ):
        logger.__init__(self, "data_table :: ", verbose)
        self.name = name or "untitled {}".format(data_table.TABLE_COUNT)
        data_table.TABLE_COUNT += 1
        if headers is not None:
            self.headers = headers
            data = np.asarray(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, 1)
            self.data = data
        elif isinstance(data, pd.DataFrame):
            self.headers = data.columns
            self.data = data
        elif isinstance(data, data_table):
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
    
    def norm(
            self,
            data=None,
            norm_type=0,
            out_name=None,
            rng=None,
            **scaler_args
    ):
        
        if rng is not None:
            return self.norm_alt(rng, out_name)
        
        if isinstance(norm_type, str):
            norm_type = getattr(self.NORM_TYPES, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NORM_TYPES, self._RDICT_NORM_TYPES[norm_type])
        
        assert isinstance(norm_type, self.NORM_TYPES)
        
        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, data_table), "data must be data_table type"
        
        if out_name is None:
            out_name = "'{}' normed to '{}'".format(data.name, self.name)
        
        ret = data_table(pd.DataFrame(self.scaler.transform(data.df), columns=data.df.columns, index=data.df.index),
                         name=out_name)
        return ret
    
    def inorm(
            self,
            data=None,
            norm_type=0,
            out_name=None,
            rng=None,
            **scaler_args
    ):
        if rng is not None:
            return self.inorm_alt(rng, out_name)
        if isinstance(norm_type, str):
            norm_type = getattr(self.NORM_TYPES, norm_type)
        elif isinstance(norm_type, int):
            norm_type = getattr(self.NORM_TYPES, self._RDICT_NORM_TYPES[norm_type])
        
        assert isinstance(norm_type, self.NORM_TYPES)
        
        self.scaler = getattr(prep, norm_type.name)(**scaler_args)
        self.scaler.fit(self.df)
        
        if data is None:
            data = self
        
        assert isinstance(data, data_table), "data must be data_table type"
        
        if out_name is None:
            out_name = "'{}' inv_normed to '{}'".format(data.name, self.name)
        
        ret = data_table(
            pd.DataFrame(self.scaler.inverse_transform(data.df), columns=data.df.columns, index=data.df.index),
            name=out_name)
        
        # ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
        return ret
    
    def norm_alt(self, rng, out_name=None):
        if out_name is None:
            out_name = "{} norm".format(self.name)
        
        return data_table((self.df - rng[:, 0]) / (rng[:, 1] - rng[:, 0]), name=out_name)
    
    def inorm_alt(self, rng, out_name=None):
        
        if out_name is None:
            if self.name.endswith('norm'):
                out_name = self.name.replace('norm', '').strip()
            else:
                out_name = "{} inverse normed".format(self.name)
        
        ret = data_table(self.df * (rng[:, 1] - rng[:, 0]) + rng[:, 0], name=out_name)
        
        # ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
        return ret
    
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
        
        return data_table(t1, headers=match_list, name=self.name), data_table(t2, headers=other, name=self.name)
    
    def train_test_split(self, test_fraction=0.25, random_state=None):
        dtrain, dtest = train_test_split(self, test_size=test_fraction, random_state=random_state)
        return (data_table(dtrain, name="train"),
                data_table(dtest, name="test"))
    
    def split_by_event(self, test_fraction=0.25, random_state=None, n_skip=2):
        # shuffle event indicies
        train_idx, test_idx = train_test_split(self.df.index[0::n_skip], test_size=test_fraction,
                                               random_state=random_state)
        train, test = [np.asarray([x + i for i in range(n_skip)]).T.flatten() for x in [train_idx, test_idx]]
        return (data_table(self.df.loc[train], name="train"),
                data_table(self.df.loc[test], name="test"))
    
    def plot(
            self,
            others=[],
            values="*",
            bins=32,
            rng=None,
            cols=4,
            ticksize=8,
            fontsize=10,
            normed=0,
            figloc="lower right",
            figsize=16,
            alpha=0.7,
            xscale="linear",
            yscale="linear",
            histtype='step',
            figname="Untitled",
            savename=None,
    ):
        if isinstance(values, str):
            values = [key for key in self.headers if glob.fnmatch.fnmatch(key, values)]
        if not hasattr(values, "__iter__"):
            values = [values]
        for i in range(len(values)):
            if isinstance(values[i], int):
                values[i] = self.headers[values[i]]
        
        if not isinstance(others, list) or isinstance(others, tuple):
            others = [others]
        
        for i in range(len(others)):
            if not isinstance(others[i], data_table):
                others[i] = data_table(others[i], headers=self.headers)
        
        n = len(values)
        rows = self._rows(cols, n)
        
        if n < cols:
            cols = n
            rows = 1
        
        plot_data = [self[v] for v in values]
        plot_others = [[other[v] for v in values] for other in others]
        
        if rng is None:
            rmax = np.max([d.max().values for d in ([self] + others)], axis=0)
            rmin = np.min([d.min().values for d in ([self] + others)], axis=0)
            rng = np.array([rmin, rmax]).T
        elif len(rng) == 2 and all([not hasattr(r, "__iter__") for r in rng]):
            rng = [rng for i in range(len(plot_data))]
        
        weights = None
        
        if not isinstance(figsize, tuple):
            figsize = (figsize, rows * float(figsize) / cols)
        
        self.log("plotting distrubution(s) for table(s) {}".format([self.name, ] + [o.name for o in others]))
        plt.rcParams['figure.figsize'] = figsize
        
        use_weights = False
        if normed == 'n':
            normed = 0
            use_weights = True
        
        for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            if use_weights:
                weights = np.ones_like(plot_data[i]) / float(len(plot_data[i]))
            
            ax.hist(plot_data[i], bins=bins, range=rng[i], histtype=histtype, normed=normed, label=self.name,
                    weights=weights, alpha=alpha)
            
            for j in range(len(others)):
                if use_weights:
                    weights = np.ones_like(plot_others[j][i]) / float(len(plot_others[j][i]))
                
                # ax.hist(plot_others[j][i]/plot_data[i].shape[0], bins=bins, range=rng[i], histtype=histtype, label=others[j].name, normed=0, weights=weights, alpha=alpha)
                ax.hist(plot_others[j][i], bins=bins, range=rng[i], histtype=histtype, label=others[j].name,
                        normed=normed, weights=weights, alpha=alpha)
            
            plt.xlabel(plot_data[i].name + " {}-scaled".format(xscale), fontsize=fontsize)
            plt.ylabel("{}-scaled".format(yscale), fontsize=fontsize)
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.gca().spines['left']._adjust_location()
            plt.gca().spines['bottom']._adjust_location()
        
        handles, labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc=figloc)
        plt.suptitle(figname)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename)
    
    
    def cdrop(self, globstr, inplace=False):
        to_drop = list(utils.parse_globlist(globstr, list(self.df.columns)))
        
        if inplace:
            modify = self
        else:
            ret = data_table(self)
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
            ret = data_table(self)
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
        return data_table(self.df.join(other.df), name=out_name)
    
    def _rows(self, cols, n):
        return n / cols + bool(n % cols)
    
    def split_to_jets(self):
        return utils.split_to_jets(self)