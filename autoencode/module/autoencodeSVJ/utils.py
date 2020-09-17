# import autoencodeSVJ.trainer

import numpy as np
from collections import OrderedDict as odict
from operator import mul
import h5py
import glob
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep
import pandas as pd
import os
import traceback
import matplotlib.pyplot as plt
import glob
import pandas as pd
import prettytable
#from StringIO import StringIO
from enum import Enum
import subprocess


from functools import reduce


import chardet

plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams.update({'font.size': 18})

class logger:
    """
    good logging parent class.
    Usage:
        When creating a class, subclass this class. 
        You can override the _LOG_PREFIX, _ERROR_PREFIX,
        and VERBOSE member variables either in the derived
        class __init__, or when calling the logger __init__.

        Ex:
        class <classname>(logger):
            def __init__(self, ..., verbose):
                ...
                logger.__init__(self, VERBOSE=verbose, LOG_PREFIX="MY class log prefix")
                ...
            OR
            def __init__(self, ..., verbose):
                logger.__init__(self)
                self.VERBOSE = verbose
                self._LOG_PREFIX = "my log prefix: "

        From there, you can use the logging functions as members of your class.
        i.e. self.log("log message") or self.error("log message")
        
        To get the log as a string, pass the 'string=True' argument to 
        either of the 'log' or 'error' member functions
                
    """


    def __init__(
        self,
        LOG_PREFIX = "logger :: ",
        ERROR_PREFIX = "ERROR: ",
        VERBOSE = True,
    ):
        self._LOG_PREFIX = LOG_PREFIX
        self._ERROR_PREFIX = ERROR_PREFIX
        self.VERBOSE = VERBOSE

    def log(self, s, string=False):
        if string:
            if self.VERBOSE:
                return self._log_str(s, self._LOG_PREFIX)
            return ''
        if self.VERBOSE: 
            self._log_base(s, self._LOG_PREFIX)

    def error(self, s, string=False):
        if string:
            return self._log_str(s, self._LOG_PREFIX + self._ERROR_PREFIX)    
        self._log_base(s, self._LOG_PREFIX + self._ERROR_PREFIX)

    def _log_base(self, s, prefix):
        if isinstance(s, str):
            for line in s.split('\n'):
                print((prefix + str(line)))
        else:
            print((prefix + str(s)))

    def _log_str(self, s, prefix):
        out = ''
        if isinstance(s, str):
            for line in s.split('\n'):
                out += prefix + str(line) + '\n'
        else:
            out += prefix + str(line) + '\n'
        return out

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
        if  isinstance(self.data, pd.DataFrame):
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
            out_name = "'{}' normed to '{}'".format(data.name,self.name)

        ret = data_table(pd.DataFrame(self.scaler.transform(data.df), columns=data.df.columns, index=data.df.index), name=out_name)
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
            out_name = "'{}' inv_normed to '{}'".format(data.name,self.name)

        ret = data_table(pd.DataFrame(self.scaler.inverse_transform(data.df), columns=data.df.columns, index=data.df.index), name=out_name)

        # ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
        return ret
        
    def norm_alt(
        self,
        rng,
        out_name=None,
    ):
        
        if out_name is None:
            out_name = "{} norm".format(self.name)


        return data_table((self.df - rng[:,0])/(rng[:,1] - rng[:,0]), name=out_name)

    def inorm_alt(
        self,
        rng,
        out_name=None
    ):

        if out_name is None:
            if self.name.endswith('norm'):
                out_name = self.name.replace('norm', '').strip()
            else:
                out_name = "{} inverse normed".format(self.name)

        ret = data_table(self.df*(rng[:,1] - rng[:,0]) + rng[:,0], name=out_name)

        # ret = data_table(self.scaler.inverse_transform(data.df), headers=self.headers, name=out_name)
        return ret     

    def __getattr__(
        self,
        attr,
    ):
        if hasattr(self.df, attr):
            return self.df.__getattr__(attr)
        else:
            raise AttributeError

    def __getitem__(self, item):
        return self.df[item]

    def __str__(
        self,
    ):
        return self.df.__str__()

    def __repr__(
        self,
    ):
        return self.df.__repr__()

    def split_by_column_names(
        self,
        column_list_or_criteria,
    ):
        match_list = None
        if isinstance(column_list_or_criteria, str):
            match_list = [c for c in self.headers if glob.fnmatch.fnmatch(c, column_list_or_criteria)]
        else:
            match_list = list(column_list_or_criteria)

        other = [c for c in self.headers if c not in match_list]

        t1,t2 = self.df.drop(other,axis=1),self.df.drop(match_list,axis=1)

        
        return data_table(t1, headers=match_list, name=self.name), data_table(t2, headers=other, name=self.name)
        
    def train_test_split(
        self,
        test_fraction=0.25,
        random_state=None,
    ):
        dtrain, dtest = train_test_split(self, test_size=test_fraction, random_state=random_state)
        return (data_table(dtrain, name="train"),
            data_table(dtest, name="test"))

    def split_by_event(
        self,
        test_fraction=0.25,
        random_state=None,
        n_skip=2,
    ):
        # shuffle event indicies
        train_idx, test_idx = train_test_split(self.df.index[0::n_skip], test_size=test_fraction, random_state=random_state)
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
            rng =  np.array([rmin, rmax]).T
        elif len(rng) == 2 and all([not hasattr(r, "__iter__") for r in rng]):
            rng = [rng for i in range(len(plot_data))]

        weights = None

        if not isinstance(figsize, tuple):
            figsize = (figsize, rows*float(figsize)/cols)

        self.log("plotting distrubution(s) for table(s) {}".format([self.name,] + [o.name for o in others]))
        plt.rcParams['figure.figsize'] = figsize

        use_weights = False
        if normed == 'n':
            normed = 0
            use_weights = True

        for i in range(n):
            ax = plt.subplot(rows, cols, i + 1)
            if use_weights:
                weights = np.ones_like(plot_data[i])/float(len(plot_data[i]))
            
            ax.hist(plot_data[i], bins=bins, range=rng[i], histtype=histtype, normed=normed, label=self.name, weights=weights, alpha=alpha)
            
            for j in range(len(others)):
                if use_weights:
                    weights = np.ones_like(plot_others[j][i])/float(len(plot_others[j][i]))

                # ax.hist(plot_others[j][i]/plot_data[i].shape[0], bins=bins, range=rng[i], histtype=histtype, label=others[j].name, normed=0, weights=weights, alpha=alpha)
                ax.hist(plot_others[j][i], bins=bins, range=rng[i], histtype=histtype, label=others[j].name, normed=normed, weights=weights, alpha=alpha)

            plt.xlabel(plot_data[i].name + " {}-scaled".format(xscale), fontsize=fontsize)
            plt.ylabel("{}-scaled".format(yscale), fontsize=fontsize)
            plt.xticks(size=ticksize)
            plt.yticks(size=ticksize)
            plt.yscale(yscale)
            plt.xscale(xscale)
            plt.gca().spines['left']._adjust_location()
            plt.gca().spines['bottom']._adjust_location()


        handles,labels = ax.get_legend_handles_labels()
        plt.figlegend(handles, labels, loc=figloc)
        plt.suptitle(figname)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename)
    
    def cdrop(
        self,
        globstr,
        inplace=False,
    ):
        to_drop = list(parse_globlist(globstr, list(self.df.columns)))
        modify = None
        if inplace:
            modify = self
        else:
            ret = data_table(self)
            modify = ret
            
        for i, d in enumerate(to_drop):
            if type(d) is str:
                to_drop[i] = d.encode("utf-8")
        modify.df.drop(to_drop, axis=1, inplace=True)
        
        modify.headers = list(modify.df.columns)
        modify.data = np.asarray(modify.df)
        return modify

    def cfilter(
        self, 
        globstr,
        inplace=False,
    ):
        to_keep = parse_globlist(globstr, list(self.df.columns))
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

    def cmerge(
        self,
        other,
        out_name,
    ):
        assert self.shape[0] == other.shape[0], 'data tables must have same number of samples'
        return data_table(self.df.join(other.df), name=out_name)

    def _rows(
        self,
        cols,
        n
    ):
        return n/cols + bool(n%cols)

    def split_to_jets(
        self,
    ):
        return split_to_jets(self)

class data_loader(logger):
    """
    data loader / handler/merger for h5 files with the general format
    of this repository
    """

    def __init__(
        self,
        name,
        verbose=True
    ):
        logger.__init__(self)
        self.name = name
        self._LOG_PREFIX = "data_loader :: "
        self.VERBOSE = verbose
        self.samples = odict()
        self.sample_keys = None
        self.data = odict()
        self.labels = odict()

    def add_sample(
        self,
        sample_path,
    ):
        filepath = smartpath(sample_path)        
        
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

    def make_table(
        self,
        key,
        name=None,
        third_dim_handle="stack", # stack, combine, or split
    ):
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
            if third_dim_handle=="split":
                for i,(r,t) in enumerate(zip(ret, table)):
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
        splits = [0,] + [sum(sizes[:i+1]) for i in range(len(sizes))]

        dataset = np.empty((sample_size, sum(sizes)))

        for i, datum in enumerate(data_dict.values()):
            dataset[:,splits[i]:splits[i + 1]] = datum.reshape(datum.shape[0], sizes[i])
        # self.log("Dataset shape: {0}".format(dataset.shape))
        return dataset, data_dict

    def save(
        self,
        filepath,
        force=False,
    ):
        """saves the current sample sets as one h5 file to the filepath specified"""
        filepath = smartpath(filepath)
        if not filepath.endswith('.h5'):
            filepath += '.h5'

        if os.path.exists(filepath) and not force:
            self.error("Path '{0}' already contains data. Use the 'force' argument to force saving".format(filepath))
            return 1

        f = h5py.File(filepath, "w")
        for key,data in list(self.data.items()):
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

def parse_globlist(glob_list, match_list):
    if not hasattr(glob_list, "__iter__") or isinstance(glob_list, str):
        glob_list = [glob_list]
    
    for i,x in enumerate(glob_list):
        if isinstance(x, int):
            glob_list[i] = match_list[x]
        
    assert all([isinstance(c, str) for c in glob_list])

    match = set()
    
    if type(match_list[0]) is bytes:
        match_list = [x.decode("utf-8") for x in match_list]
    
    for g in glob_list:
        match.update(glob.fnmatch.filter(match_list, g))

    return match

delphes_jet_tags_dict = {
    1: "down",
    2: "up",
    3: "strange",
    4: "charm",
    5: "bottom",
    6: "top",
    21: "gluon",
    9: "gluon"
}

def plot_error_ratios(main_error, compare_errors, metric='mse', bins= 40, log=False, rng=None, alpha=0.6):
    import matplotlib.pyplot as plt
    raw_counts, binned = np.histogram(main_error[metric], bins=bins, normed=False, range=rng)
    raw_counts = raw_counts.astype(float)
    
    zeros = np.where(raw_counts == 0)[0]
    if len(zeros) > 0:
        cutoff_index = zeros[0]
    else:
        cutoff_index = len(raw_counts)
    raw_counts = raw_counts[:cutoff_index]
    binned = binned[:cutoff_index + 1]
        
    ratios = []
    for e in compare_errors:
        counts, _ = np.histogram(e[metric], bins=bins, normed=False, range=rng)
        counts = counts.astype(float)[:cutoff_index]
        ratio = counts/raw_counts
        ratio_plot = ratio*(main_error.shape[0]/e.shape[0])
        ratios.append((ratio, raw_counts, counts))
        toplot = np.asarray(list(ratio_plot) + [0])
        err = np.asarray(list(1/counts) + [0])
        plt.plot(binned, toplot, label=e.name.lstrip("error ") + " ({0})".format(e.shape[0]), marker='o', alpha=alpha)
        if log:
            plt.yscale("log")
    plt.legend()
    plt.show()
    return ratios

def get_errors(true, pred, out_name="errors", functions=["mse", "mae"], names=[None, None], index=None):
    import tensorflow as tf
    import keras
    if names is None:
        names = ['err {}'.format(i) for i in range(len(functions))]
    
    functions_keep = []
    for i,f in enumerate(functions):
        if isinstance(f, str):
            fuse = getattr(keras.losses, f)
            functions_keep.append(fuse)
            names[i] = f
        else:
            functions_keep.append(f)
    
    raw = [func(true, pred) for func in functions_keep]
    raw = np.asarray([keras.backend.eval(x) if isinstance(x, tf.Tensor) else x for x in raw]).T
    return data_table(
        pd.DataFrame(raw, columns=[str(f) for f in names], index=index),
        name=out_name
    )

def split_table_by_column(column_name, df, tag_names=None, keep_split_column=False, df_to_write=None):
    if df_to_write is None:
        df_to_write = df
    tagged = []
    unique = set(df.loc[:,column_name].values)
    if tag_names is None:
        tag_names = dict([(u, str(u)) for u in unique])

    if isinstance(df_to_write, pd.Series):
        df_to_write = pd.DataFrame(df_to_write)

    assert df.shape[0] == df_to_write.shape[0], 'writing and splitting dataframes must have the same size!'

    df = df.copy().reset_index(drop=True)
    df_to_write = df_to_write.copy().reset_index(drop=True)
    
    gb = df.groupby(column_name)
    index = gb.groups
    
    for region, idx in list(index.items()):
        if keep_split_column or column_name not in df_to_write:
            tagged.append(data_table(df_to_write.iloc[idx], headers=list(df_to_write.columns), name=tag_names[region]))
        else:
            tagged.append(data_table(df_to_write.iloc[idx].drop(column_name, axis=1), name=tag_names[region]))
    return tagged, dict([(tag_names[k], v) for k,v in list(index.items())])

def smartpath(path):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)

def get_cutflow_table(glob_path):
    paths = glob.glob(glob_path)
    assert len(paths) > 0, "must have SOME paths"

    ret = odict()
    for path in paths:
        with open(path) as f:
            values_comp, keys_comp = [x.strip('\n').split(',') for x in f.readlines()]
            values_comp = list(map(int, values_comp))
            keys_comp = list(map(str.strip, ['no cut'] + keys_comp))
            for k,v in zip(keys_comp, values_comp):
                if k not in ret:
                    ret[k] = 0
                ret[k] = ret[k] + v
    df = pd.DataFrame(list(ret.items()), columns=['cut_name', 'n_events'])
    df['abs eff.'] = np.round(100.*(df.n_events / df.n_events[0]), 2)
    df['rel eff.'] = np.round([100.] + [100.*(float(df.n_events[i + 1]) / float(df.n_events[i])) for i in range(len(df.n_events) - 1)], 2)
    
    return df

def get_training_data(glob_path, verbose=1):
    paths = glob.glob(glob_path)
    d = data_loader("main sample", verbose=verbose)
    for p in paths:
        d.add_sample(p)
    tables = []
    
    return d.make_table("data", "*features_data", "*features_names") 

def get_training_data_jets(glob_path, verbose=1):
    return split_to_jets(get_training_data(glob_path, verbose))

def get_subheaders(data):
    classes = {}
    i = 0
    n = 0
    h = data.headers
    while i < len(h):
        if str(n) not in h[i]:
            n += 1
            continue
        rep = h[i]
        if "j{}".format(n) in rep:
            rep = rep.replace("j{}".format(n), "jet")
        elif "jet{}".format(n) in rep:
            rep = rep.replace("jet{}".format(n), "jet")
        if n not in classes:
            classes[n] = []
        classes[n].append(rep)
        i += 1
    return classes

def get_selections_dict(list_of_selections):
    ret = {}
    for sel in list_of_selections:
        with open(sel, 'r') as f:
            data = [x.strip('\n') for x in f.readlines()]
        for elt in data:
            key, raw = elt.split(': ')
            ret[key] = list(map(int, raw.split()))
    return ret

def get_repo_info():
    info = {}
    info['head'] = subprocess.Popen("git rev-parse --show-toplevel".split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
    info['name'] = subprocess.Popen("git config --get remote.origin.url".split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode("utf-8").strip('\n')
    return info

def split_to_jets(data):
    """
    given a data table with values for the n leading jets, split into one data 
    table for all jets.
    """
    headers = get_subheaders(data)
    assert len(set().union(*list(headers.values()))) == len(list(headers.values())[0])
    jets = []
    next = data
    for h in headers:
        to_add, next = next.split_by_column_names("jet{}*".format(h))
        if to_add.shape[1] == 0:
            to_add, next = next.split_by_column_names("j{}*".format(h))
        jets.append(
            data_table(
                data=np.asarray(to_add),
                headers=headers[h],
                name="jet {}".format(h)
            )
        )

    full = data_table(
        data=np.vstack([jt.df for jt in jets]),
        headers=jets[0].headers,
        name="all jet data"
    )
    return full, jets

def log_uniform(low, high, size=None, base=10.):
    return float(base)**(np.random.uniform(np.log(low)/np.log(base), np.log(high)/np.log(base), size))

def split_by_tag(data, tag_column="jetFlavor", printout=True):
    tagged, tag_index = split_table_by_column(
        tag_column,
        data,
        delphes_jet_tags_dict,
        False
    )
    if printout:
        sizes = [x.shape[0] for x in tagged]
        for t,s in zip(tagged, sizes):
            print(("{} jet: {}, {}%".format(t.name, s, round(100.*s/sum(sizes), 1))))
        
    return tagged, tag_index
    
def compare_tags(datasets):
    
    tags = [dict([(t.name, t) for t in split_by_tag(x, printout=False)[0]]) for x in datasets]
    tag_ids = set().union(*[set([tn for tn in tlist]) for tlist in tags])
    
    for tag_id in tag_ids:
        print(("{}:".format(tag_id)))
        for t,d in zip(tags, datasets):
            
            if tag_id in t:
                tag = t[tag_id]
                print(("\t{:.1f}% ({}) {}".format(100.*tag.shape[0]/d.shape[0], tag.shape[0], d.name)))
            
def get_recon_errors(data_list, autoencoder, **kwargs):

    if not isinstance(data_list, list):
        data_list = [data_list]
    
    recon = []
    errors = []
    
    for i,d in enumerate(data_list):
        recon.append(
            data_table(
                pd.DataFrame(autoencoder.predict(d.data), columns=d.columns, index=d.index),
                name="{0} pred".format(d.name)
            )
        )
        errors.append(
            get_errors(recon[i].data, d.data, out_name="{0} error".format(d.name), index=d.df.index, **kwargs)
        )
        
    return errors, recon

def roc_auc_dict(data_errs, signal_errs, metrics=['mse', 'mae'], *args, **kwargs):
    from sklearn.metrics import roc_curve, roc_auc_score
    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(signal_errs, list):
        signal_errs = [signal_errs]

    if not isinstance(data_errs, list):
        data_errs = [data_errs]
    
    if len(data_errs) == 1:
        data_errs = [data_errs[0] for i in range(len(signal_errs))]

    ret = {}    
    
    for i,(data_err,signal_err) in enumerate(zip(data_errs, signal_errs)):
        
        ret[signal_err.name] = {}
        
        for j,metric in enumerate(metrics):
            ret[signal_err.name][metric] = {} 
            pred = np.hstack([signal_err[metric].values, data_err[metric].values])
            true = np.hstack([np.ones(signal_err.shape[0]), np.zeros(data_err.shape[0])])

            roc = roc_curve(true, pred)
            auc = roc_auc_score(true, pred)
            
            ret[signal_err.name][metric]['roc'] = roc
            ret[signal_err.name][metric]['auc'] = auc

    return ret

def roc_auc_plot(data_errs, signal_errs, metrics='loss', *args, **kwargs):
    from sklearn.metrics import roc_curve, roc_auc_score
    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(signal_errs, list):
        signal_errs = [signal_errs]

    if not isinstance(data_errs, list):
        data_errs = [data_errs]
    
    if len(data_errs) == 1:
        data_errs = [data_errs[0] for i in range(len(signal_errs))]
        
    fig, ax_begin, ax_end, plt_end, colors = get_plot_params(1, *args, **kwargs)
    ax = ax_begin(0)
    styles = [ '-','--','-.',':']
    for i,(data_err,signal_err) in enumerate(zip(data_errs, signal_errs)):
        
        for j,metric in enumerate(metrics):
            pred = np.hstack([signal_err[metric].values, data_err[metric].values])
            true = np.hstack([np.ones(signal_err.shape[0]), np.zeros(data_err.shape[0])])

            roc = roc_curve(true, pred)
            auc = roc_auc_score(true, pred)
        
            ax.plot(roc[0], roc[1], styles[j%len(styles)], c=colors[i%len(colors)], label='{} {}, AUC {:.4f}'.format(signal_err.name, metric, auc))

    ax.plot(roc[0], roc[0], '--', c='black')
    ax_end("false positive rate", "true positive rate")
    plt_end()
    plt.show()
    
def OLD_load_all_data(data_path, name, cols_to_drop = ["jetM", "*MET*", "*Delta*"]):
    """Returns: a tuple with... 
        a data_table with all data, (columns dropped),
        a list of data_tables, by jetFlavor tag (columns dropped),
        a list of data_tables, split by jet # (columns dropped)
    """
    repo_head = get_repo_info()['head']

    if not os.path.exists(data_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        data_path = os.path.join(repo_head, data_path)
    
    # if not os.path.exists(signal_path):
    #     assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
    #     signal_path = os.path.join(repo_head, signal_path)
        
    # if not os.path.exists(model_path):
    #     assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
    #     model_path = os.path.join(repo_head, model_path)

    data, data_jets = get_training_data_jets(data_path, 0)

    for i,d in enumerate(data_jets):
        data_jets[i].name = d.name + " " + name

    # signal, signal_jets = get_training_data_jets(signal_path, 0)

    data.name = name

    data.cdrop(cols_to_drop, inplace=True)
    # signal.cdrop(cols_to_drop, inplace=True)
    
    print(("{} tags:".format(name)))
    print("")
    tagged_data, tag_index_data = split_by_tag(data)
    print("")
    # print("signal tags:")
    # print("")
    # tagged_signal, tag_index_signal = split_by_tag(signal)
    # print("")

    return data, tagged_data, data_jets

def evaluate_model(data_path, signal_path, model_path):
    # get h5 datasets
    repo_head = get_repo_info()['head']

    if not os.path.exists(data_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        data_path = os.path.join(repo_head, data_path)
    
    if not os.path.exists(signal_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        signal_path = os.path.join(repo_head, signal_path)
        
    if not os.path.exists(model_path):
        assert len(repo_head) > 0, "not running jupyter notebook from within a repo!! Prolly won't work :-)"
        model_path = os.path.join(repo_head, model_path)

        
    data, data_jets = get_training_data_jets(data_path, 0)
    signal, signal_jets = get_training_data_jets(signal_path, 0)

    signal.name = "signal"
    data.name = "background"

    # drop some unused columns
    cols_to_drop = ["jetM", "*MET*", "*Delta*"]

    data.cdrop(cols_to_drop, inplace=True)
    signal.cdrop(cols_to_drop, inplace=True)
    
    print("data tags:")
    print("")
    tagged_data, tag_index_data = split_by_tag(data)
    print("")
    print("signal tags:")
    print("")
    tagged_signal, tag_index_signal = split_by_tag(signal)
    print("")
    
    data_raw = data.cdrop("*Flavor")
    signal_raw = signal.cdrop("*Flavor")

    norm_args = {
        "norm_type": "StandardScaler",
    #     "feature_range": (0.01, 0.99)
    }

    data_norm = data_raw.norm(**norm_args)
    signal_norm = data_raw.norm(signal_raw, **norm_args)
    

    instance = trainer.trainer(model_path, verbose=False)
    autoencoder = instance.load_model()
    encoder, decoder = autoencoder.layers[1:]
    
    tagged_norm = [data_raw.norm(t, **norm_args) for t in tagged_data]

    errors, recon = get_recon_errors([data_norm, signal_norm] + tagged_norm, autoencoder)
    
    data_recon, signal_recon, tagged_recon = recon[0], recon[1], recon[2:]
    data_error, signal_error, tagged_error = errors[0], errors[1], errors[2:]
    
    roc_auc_plot(data_error, signal_error)
    
    data_recon.name = "background pred"
    signal_recon.name = "signal pred"
    data_norm.name = "background (norm)"
    signal_norm.name = "signal (norm)"
    
    data_norm.plot(
        [data_recon, signal_norm, signal_recon],
        normed=1, bins=70, cols=2, figsize=(10,20),
#         rng=((0,2), (-3.2, 3.2), (0,2500), (0,1), (0, 0.8),(0,.1), (0,3000)),
        figname="signal/data comparison"
    )
    
    signal_error.plot(data_error, normed=1, bins=100, figname="signal vs data errors")

def percentile_normalization_ranges(data, n):
    return np.asarray(list(zip(np.percentile(data, n, axis=0), np.percentile(data, 100-n, axis=0))))

def get_plot_params(
    n_plots,
    cols=4,
    figsize=20.,
    yscale='linear',
    xscale='linear',
    figloc='lower right',
    figname='Untitled',
    savename=None,
    ticksize=8,
    fontsize=5,
    colors=None
):
    rows =  n_plots/cols + bool(n_plots%cols)
    if n_plots < cols:
        cols = n_plots
        rows = 1
        
    if not isinstance(figsize, tuple):
        figsize = (figsize, rows*float(figsize)/cols)
    
    fig = plt.figure(figsize=figsize)
    
    def on_axis_begin(i):
        return plt.subplot(rows, cols, i + 1)    
    
    def on_axis_end(xname, yname=''):
        plt.xlabel(xname + " ({0}-scaled)".format(xscale))
        plt.ylabel(yname + " ({0}-scaled)".format(yscale))
        plt.xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.gca().spines['left']._adjust_location()
        plt.gca().spines['bottom']._adjust_location()
        
    def on_plot_end():
        handles,labels = plt.gca().get_legend_handles_labels()
        by_label = odict(list(zip(list(map(str, labels)), handles)))
        plt.figlegend(list(by_label.values()), list(by_label.keys()), loc=figloc)
        # plt.figlegend(handles, labels, loc=figloc)
        plt.suptitle(figname)
        plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01, rect=[0, 0.03, 1, 0.95])
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename)
            
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(colors) < n_plots:
        print("too many plots for specified colors. overriding with RAINbow")
        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, n_plots)) 
    return fig, on_axis_begin, on_axis_end, on_plot_end, colors

def plot_spdfs(inputs, outputs, bins=100, *args, **kwargs):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]
        
    # assert all([isinstance(inp, data_table) for inp in inputs]), "inputs mst be utils.data_table format"
    assert len(inputs) > 0, "must have SOME inputs"
    assert len(outputs) > 0, "must have SOME outputs"
    assert len(inputs) == len(outputs), "# of outputs and inputs must be the same"
    
    columns = inputs[0].headers
    assert all([columns == inp.headers for inp in inputs]), "all inputs must have identical column titles"

    fig, ax_begin, ax_end, plt_end, colors = get_plot_params(len(columns), *args, **kwargs)
    
    for i,name in enumerate(columns):
        
        ax_begin(i)

        for j, (IN, OUT) in enumerate(zip(inputs, outputs)):

            dname = IN.name
            centers, (content, content_new), width = get_bin_content(IN.data[:,i], OUT[0][:,i], OUT[1][:,i], bins) 

            sfac = float(IN.shape[0])
            plt.errorbar(centers, content/sfac, xerr=width/2., yerr=np.sqrt(content)/sfac, fmt='.', c=colors[j], label='{} input'.format(dname))
            plt.errorbar(centers, content_new/sfac, xerr=width/2., fmt='--', c=colors[j], label='{} spdf'.format(dname), alpha=0.7)
    #     plt.hist(mu, histtype='step', bins=bins)

        ax_end(name)
        
    plt_end()

def get_bin_content(aux, mu, sigma, bins=50):
    
    hrange = (np.percentile(aux, 0.1), np.percentile(aux, 99.9))
    
    content, edges = np.histogram(aux, bins=bins, range=hrange)
    centers = 0.5*(edges[1:] + edges[:-1])
    
    width = centers[1] - centers[0]
    
    bin_content = np.sum(content)*width*sum_of_gaussians(centers, mu, sigma)
    
    return centers, (content, bin_content), width

def sum_of_gaussians(x, mu_vec, sigma_vec):
    x = np.atleast_2d(x)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    x_norm = (x - mu_vec)/sigma_vec
    single_gaus_val = np.exp(-0.5*np.square(x_norm))/(sigma_vec*np.sqrt(2*np.pi))
    return np.sum(single_gaus_val, axis=1)/mu_vec.shape[0]

def glob_in_repo(globstring):
    repo_head = get_repo_info()['head']
    files = glob.glob(os.path.abspath(globstring))
    
    if len(files) == 0:
        files = glob.glob(os.path.join(repo_head, globstring))
    
    return files


def all_modify(tables, hlf_to_drop=['Energy', 'Flavor']):
    if not isinstance(tables, list) or isinstance(tables, tuple):
        tables = [tables]
    for i, table in enumerate(tables):
        tables[i].cdrop(['0'] + hlf_to_drop, inplace=True)
        
        newNames = dict()
        
        for column in table.df.columns:
            encoding = chardet.detect(column)["encoding"]
            if column.isdigit():
                
                newNames[column] = "eflow %s" % (column.decode(encoding))
            elif type(column) is bytes:
                newNames[column] = column.decode(encoding)
        
        tables[i].df.rename(columns=newNames, inplace=True)
        tables[i].headers = list(tables[i].df.columns)
    if len(tables) == 1:
        return tables[0]
    return tables

def hlf_modify(tables, hlf_to_drop=['Energy', 'Flavor']):
    if not isinstance(tables, list) or isinstance(tables, tuple):
        tables = [tables] 
    for i,table in enumerate(tables):
        tables[i].cdrop(hlf_to_drop, inplace=True)
    if len(tables) == 1:
        return tables[0]
    return tables

def eflow_modify(tables):
    if not isinstance(tables, list) or isinstance(tables, tuple):
        tables = [tables] 
    for i,table in enumerate(tables):
        tables[i].cdrop(['0'], inplace=True)
        tables[i].df.rename(columns=dict([(c, "eflow {}".format(c)) for c in tables[i].df.columns if c.isdigit()]), inplace=True)
        tables[i].headers = list(tables[i].df.columns)
    if len(tables) == 1:
        return tables[0]
    return tables

def jet_flavor_check(flavors):
    d = split_table_by_column("Flavor", flavors, tag_names=delphes_jet_tags_dict)[1]
    print((flavors.name.center(30)))
    print(("-"*30))
    for name,index in list(d.items()):
        tp = "{}:".format(name).rjust(10)
        tp = tp + "{}".format(len(index)).rjust(10)
        tp = tp + "({} %)".format(round(100.*len(index)/len(flavors), 1)).rjust(10)
        print(tp)
    print()

def jet_flavor_split(to_split, ref=None):
    if ref is None:
        ref = to_split
    return split_table_by_column("Flavor", ref, tag_names=delphes_jet_tags_dict, df_to_write=to_split, keep_split_column=False)[0]

def load_all_data(globstring, name, include_hlf=True, include_eflow=True, hlf_to_drop=['Energy', 'Flavor']):
    
    """returns...
        - data: full data matrix wrt variables
        - jets: list of data matricies, in order of jet order (leading, subleading, etc.)
        - event: event-specific variable data matrix, information on MET and MT etc. 
        - flavors: matrix of jet flavors to (later) split your data with
    """

    files = glob_in_repo(globstring)
    
    if len(files) == 0:
        print("\n\nERROR -- no files found in ", globstring, "\n\n")
        raise AttributeError

    to_include = []
    if include_hlf:
        to_include.append("jet_features")
    
    if include_eflow:
        to_include.append("jet_eflow_variables")
        
        
    if not (include_hlf or include_eflow):
        raise AttributeError
        
    d = data_loader(name, verbose=False)
    for f in files:
        d.add_sample(f)
        
    train_modify=None
    if include_hlf and include_eflow:
        train_modify = lambda *args, **kwargs: all_modify(hlf_to_drop=hlf_to_drop, *args, **kwargs)
    elif include_hlf:
        train_modify = lambda *args, **kwargs: hlf_modify(hlf_to_drop=hlf_to_drop, *args, **kwargs)
    else:
        train_modify = eflow_modify
        
    event = d.make_table('event_features', name + ' event features')
    data = train_modify(d.make_tables(to_include, name, 'stack'))
    jets = train_modify(d.make_tables(to_include, name, 'split'))
    flavors = d.make_table('jet_features', name + ' jet flavor', 'stack').cfilter("Flavor")
    
    return data, jets, event, flavors

def BDT_load_all_data(
    SVJ_path, QCD_path, 
    test_split=0.2, random_state=-1,
    include_hlf=True, include_eflow=True,
    hlf_to_drop=['Energy', 'Flavor']
):
    """General-purpose data loader for BDT training, which separates classes and splits data into training/testing data.
    
    Args: 
        SVJ_path (str): glob-style specification of .h5 files to load as SVJ signal
        qcd_path (str): glob-style specification of .h5 files to load as qcd background
        test_split (float): fraction of total data to use for testing 
        random_state (int): random seed, leave as -1 for random assignment
        include_hlf (bool): true to include high-level features in loaded data, false for not
        include_eflow (bool): true to include energy-flow basis features in loaded data, false for not
        hlf_to_drop (list(str)): list of high-level features to drop from the final dataset. Defaults to dropping Energy and Flavor.
    
    Returns:
        tuple(pandas.DataFrame, pandas.DataFrame): X,Y training data, where X is the data samples for each jet, and Y is the 
            signal/background tag for each jet
        tuple(pandas.DataFrame, pandas.DataFrame): X_test,Y_test testing data, where X are data samples for each jet and Y is the
            signal/background tag for each jet
    """

    if random_state < 0:
        random_state = np.random.randint(0, 2**32 - 1)

    SVJ,_,_,_ = load_all_data(SVJ_path, "SVJ", include_hlf=include_hlf, include_eflow=include_eflow, hlf_to_drop=hlf_to_drop)
    QCD,_,_,_ = load_all_data(QCD_path, "QCD", include_hlf=include_hlf, include_eflow=include_eflow, hlf_to_drop=hlf_to_drop)
    
    
    SVJ_train, SVJ_test = train_test_split(SVJ.df, test_size=test_split, random_state=random_state)
    QCD_train, QCD_test = train_test_split(QCD.df, test_size=test_split, random_state=random_state)

    SVJ_Y_train, SVJ_Y_test = [pd.DataFrame(np.ones((len(elt), 1)), index=elt.index, columns=['tag']) for elt in [SVJ_train, SVJ_test]]
    QCD_Y_train, QCD_Y_test = [pd.DataFrame(np.zeros((len(elt), 1)), index=elt.index, columns=['tag']) for elt in [QCD_train, QCD_test]]

    X = SVJ_train.append(QCD_train)
    Y = SVJ_Y_train.append(QCD_Y_train)
    
    X_test = SVJ_test.append(QCD_test)
    Y_test = SVJ_Y_test.append(QCD_Y_test)
    
    return (X, Y), (X_test, Y_test)


def get_event_index(jet_tags):
    """Get all events index ids from a list of N jet tags 
    in which all N jets originated from that event.
    """
    assert len(jet_tags) > 0
    ret = set(jet_tags[0].index)
    to_add = jet_tags[1:]
    
    for i,elt in enumerate(to_add):
        ret = ret.intersection(elt.index - i - 1)
    
    return np.sort(np.asarray(list(ret)))

def tagged_jet_dict(tags):
    """Dictionary tags
    """
    return dict(
        [
            (
                i,
                tags[tags.sum(axis=1) == i].index
            ) for i in range(tags.shape[1] + 1)
        ]
    )

def event_error_tags(
    err_jets,
    error_threshold,
    name,
    error_metric="mae",
):
    tag = [err[error_metric] > error_threshold for err in err_jets]
    tag_idx = get_event_index(tag)
    tag_data = [d.loc[tag_idx + i] for i,d in enumerate(tag)]
    jet_tags = data_table(
        pd.DataFrame(
            np.asarray(tag_data).T,
            columns=['jet {}'.format(i) for i in range(len(tag))],
            index=tag_idx/2,
        ),
        name=name + " jet tags",
    )
    return tagged_jet_dict(jet_tags)

def path_in_repo(
    filename
):
    head = get_repo_info()['head']
    suffix = ""
    comps = list(filter(len, filename.split(os.path.sep)))
    for i in range(len(comps)):
        considered = os.path.join(head, os.path.join(*comps[i:])) + suffix
        if os.path.exists(considered):
            return considered
    return None

def get_particle_PIDs_statuses(root_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import ROOT as rt
    from tqdm import tqdm

    DELPHES_DIR = os.environ["DELPHES_DIR"]
    rt.gSystem.Load("{}/lib/libDelphes.so".format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/modules/Delphes.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesClasses.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/classes/DelphesFactory.h"'.format(DELPHES_DIR))
    rt.gInterpreter.Declare('#include "{}/include/ExRootAnalysis/ExRootTreeReader.h"'.format(DELPHES_DIR))


    f = rt.TFile(root_filename)
    tree = f.Get("Delphes")
    
    parr = np.zeros((tree.Draw("Particle.PID", "", "goff"), 2))
    total = 0
    for i in tqdm(list(range(tree.GetEntries()))):
        tree.GetEntry(i)
        for p in tree.Particle:
            parr[total,:] = p.PID, p.Status
            total += 1

    df = pd.DataFrame(parr, columns=["PID", "Status"])
    new = df[abs(df.PID) > 4900100]
    counts = new.PID.value_counts()
    pdict = odict()
    for c in counts.index:
        pdict[c] = dict(new[new.PID == c].Status.value_counts())

        
    converted = pd.DataFrame(pdict).T
    converted.plot.bar(stacked=True)
    plt.show()
    return converted

def plot_particle_statuses(figsize=(7,7), **fdict):
    """With particle status name=results as the keywords, plot the particle
    statuses
    """

    cols = set().union(*[list(frame.columns) for frame in list(fdict.values())])
    parts = set().union(*[list(frame.index) for frame in list(fdict.values())])

    for name in fdict:
        fdict[name].fillna(0, inplace=True)

        for v in cols:
            if v not in fdict[name]:
                fdict[name][v] = 0
        for i in parts:
            if i not in fdict[name].index:
                fdict[name].loc[i] = 0

        fdict[name] = fdict[name][sorted(fdict[name].columns)]
        fdict[name].sort_index(inplace=True)

    for i,name in enumerate(fdict):
        ax = fdict[name].plot.bar(stacked=True, title=name, figsize=figsize)
        ax.set_xlabel("PID")
        ax.set_ylabel("Count")

        legend = ax.get_legend()
        legend.set_title("Status")
    #     plt.suptitle(name)
        plt.show()

def merge_rootfiles(glob_path, out_name, treename="Delphes"):
    import traceback as tb
    try:
        import ROOT as rt
        chain = rt.TChain(treename)
        for f in glob.glob(glob_path):
            chain.Add(f)
        chain.Merge(out_name)
        return 0
    except:
        print((tb.format_exc()))
        return 1

def set_random_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
#    K.set_session(sess)
