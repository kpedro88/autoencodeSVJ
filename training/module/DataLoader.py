import module.utils as utils
from module.Logger import Logger
from module.DataTable import DataTable

from collections import OrderedDict as odict
from sklearn.model_selection import train_test_split
import os
import h5py
import numpy as np
import pandas as pd
import chardet


class DataLoader(Logger):
    """
    data loader/handler/merger for h5 files with the general format of this repository
    """
    
    def __init__(self, name="", verbose=True):
        Logger.__init__(self)
        self.name = name
        self._LOG_PREFIX = "data_loader :: "
        self.VERBOSE = verbose
        self.samples = odict()
        self.sample_keys = None
        self.data = odict()
        self.labels = odict()

    def load_all_data(self, globstring, name, include_hlf=True, include_eflow=True, hlf_to_drop=['Energy', 'Flavor']):
    
        """returns...
            - data: full data matrix wrt variables
            - jets: list of data matricies, in order of jet order (leading, subleading, etc.)
            - event: event-specific variable data matrix, information on MET and MT etc.
            - flavors: matrix of jet flavors to (later) split your data with
        """
    
        files = utils.glob_in_repo(globstring)
    
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
    
        data_loader = DataLoader(name, verbose=False)
        for f in files:
            data_loader.add_sample(f)
    
        train_modify = None
    
        if include_hlf and include_eflow:
            train_modify = lambda *args, **kwargs: self.all_modify(hlf_to_drop=hlf_to_drop, *args, **kwargs)
        elif include_hlf:
            train_modify = lambda *args, **kwargs: self.hlf_modify(hlf_to_drop=hlf_to_drop, *args, **kwargs)
        else:
            train_modify = self.eflow_modify
    
        event = data_loader.make_table('event_features', name + ' event features')

        newNames = dict()

        for column in event.df.columns:
            if type(column) is bytes:
                encoding = chardet.detect(column)["encoding"]
                newNames[column] = column.decode(encoding)

        event.df.rename(columns=newNames, inplace=True)
        event.headers = list(event.df.columns)
        
        data = train_modify(data_loader.make_tables(to_include, name, 'stack'))
        jets = train_modify(data_loader.make_tables(to_include, name, 'split'))
        flavors = data_loader.make_table('jet_features', name + ' jet flavor', 'stack').cfilter("Flavor")
    
        return data, jets, event, flavors

    def all_modify(self, tables, hlf_to_drop=['Energy', 'Flavor']):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(['0'] + hlf_to_drop, inplace=True)
        
            newNames = dict()
        
            for column in table.df.columns:
                if type(column) is str:
                    if column.isdigit():
                        newNames[column] = "eflow %s" % (column)
                    else:
                        newNames[column] = column
                else:
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

    def hlf_modify(self, tables, hlf_to_drop=['Energy', 'Flavor']):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(hlf_to_drop, inplace=True)
        if len(tables) == 1:
            return tables[0]
        return tables

    def eflow_modify(self, tables):
        if not isinstance(tables, list) or isinstance(tables, tuple):
            tables = [tables]
        for i, table in enumerate(tables):
            tables[i].cdrop(['0'], inplace=True)
            tables[i].df.rename(columns=dict([(c, "eflow {}".format(c)) for c in tables[i].df.columns if c.isdigit()]),
                                inplace=True)
            tables[i].headers = list(tables[i].df.columns)
        if len(tables) == 1:
            return tables[0]
        return tables
    
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
            return DataTable(np.expand_dims(data, 1), headers=labels, name=name)
        elif len(data.shape) == 2:
            return DataTable(data, headers=labels, name=name)
        elif len(data.shape) == 3:
            ret = DataTable(
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
                    DataTable(
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
                return DataTable(
                    self.stack_data(data, axis=1),
                    headers=self.stack_labels(labels, data.shape[1], prefix),
                    name=name,
                )
                # combine behavior
        else:
            raise AttributeError
    
    def make_tables(self, keylist, name, third_dim_handle="stack"):
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
    
    def stack_data(self, data, axis=1):
        return np.hstack(np.asarray(np.split(data, data.shape[axis], axis=axis)).squeeze())
    
    def stack_labels(self, labels, n, prefix):
        new = []
        for j in range(n):
            for l in labels:
                new.append("{}{}_{}".format(prefix, j, l))
        return np.asarray(new)
       
   
    def _update_data(self, sample_file, keys_to_add):
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


    def BDT_load_all_data(self, qcd_path, signal_path,
                          test_split=0.2, random_state=-1,
                          include_hlf=True, include_eflow=True,
                          hlf_to_drop=['Energy', 'Flavor']):
        
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
            random_state = np.random.randint(0, 2 ** 32 - 1)
    
        # Load QCD samples
        (QCD, _, _, _) = self.load_all_data(qcd_path, "QCD",
                                                   include_hlf=include_hlf, include_eflow=include_eflow,
                                                   hlf_to_drop=hlf_to_drop)
    
        (SVJ, _, _, _) = self.load_all_data(signal_path, "SVJ",
                                                   include_hlf=include_hlf, include_eflow=include_eflow,
                                                   hlf_to_drop=hlf_to_drop)
    
        SVJ_X_train, SVJ_X_test = train_test_split(SVJ.df, test_size=test_split, random_state=random_state)
        QCD_X_train, QCD_X_test = train_test_split(QCD.df, test_size=test_split, random_state=random_state)
    
        SVJ_Y_train, SVJ_Y_test = [pd.DataFrame(np.ones((len(elt), 1)), index=elt.index, columns=['tag']) for elt in
                                   [SVJ_X_train, SVJ_X_test]]
        QCD_Y_train, QCD_Y_test = [pd.DataFrame(np.zeros((len(elt), 1)), index=elt.index, columns=['tag']) for elt in
                                   [QCD_X_train, QCD_X_test]]
    
        X_train = SVJ_X_train.append(QCD_X_train)
        Y_train = SVJ_Y_train.append(QCD_Y_train)
    
        X_test = SVJ_X_test.append(QCD_X_test)
        Y_test = SVJ_Y_test.append(QCD_Y_test)
    
        return (X_train, Y_train), (X_test, Y_test)