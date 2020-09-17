import numpy as np
from keras.models import Model, model_from_json
from collections import OrderedDict as odict
import os
import traceback
from datetime import datetime
from autoencode.module.autoencodeSVJ.utils import logger, smartpath, get_plot_params
import h5py
import matplotlib.pyplot as plt
import glob
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import keras.optimizers
import os
import collections
import pickle
import tensorflow as tf
from pathlib import Path

class pkl_file(collections.MutableMapping):
    """Dictionary which saves all attributes to a .pkl file on access/altering"""

    def __init__(self, path, verbose=1, *args, **kwargs):
        
        if not path.endswith(".pkl"):
            path += ".pkl"
        
        self.path = smartpath(path)
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

class trainer(logger):
    """
    Wraps training/testing/evaluation activity for a model in an h5 file saver, which keeps all
    training inputs/outputs, model performance stats, model weights, etc.
    """

    ### LIFE/DEATH

    def __init__(
        self,
        name,
        verbose=True,
        overwrite=False,
    ):
        logger.__init__(self)

        self._LOG_PREFIX = "train_shell :: "
        self.VERBOSE = verbose

        self.config_file = smartpath(name)
        self.path = os.path.dirname(self.config_file)

        if not self.config_file.endswith(".pkl"):
            self.config_file += ".pkl"
            
        self.config = pkl_file(self.config_file)
        
        defaults = {
            'name': name,
            'trained': False,
            'model_json': '',
            'batch_size': [],
            'epoch_splits': [],
            'metrics': {},
        }

        for k,v in list(defaults.items()):
            if k not in self.config:
                self.config[k] = v

    def _throw(
        self,
        msg,
        exc=AttributeError,
    ):
        self.close()
        self.error(msg)
        raise exc

    def close(
        self,
    ):
        try:
            del self.config
        except:
            pass

    def __del__(
        self,
    ):
        self.close()

    ### ACTUAL WRAPPERS

    def load_model(
        self,
        model=None,
        force=False,
        custom_objects=None,
    ):
        w_path = self.config_file.replace(".pkl", "_weights.h5")
        # if already trained
        if self.config['trained']:
            if self.config['model_json']:
                if model is None:
                    model = model_from_json(self.config['model_json'], custom_objects=custom_objects)
                    model.load_weights(w_path)
                    self.log("using saved model")
                else:
                    if not force:
                        model = model_from_json(self.config['model_json'], custom_objects=custom_objects)
                        model.load_weights(w_path)
                        self.error("IGNORING PASSED PARAMETER 'model'")
                        self.log("using saved model")
                    else:
                        if isinstance(model, str):
                            model = load_model(model, custom_objects=custom_objects)
                        self.log("using model passed as function argument")
            else:
                if model is None:
                    self._throw("no model passed, and saved model not found!")
                if isinstance(model, str):
                    model = load_model(model, custom_objects=custom_objects)
                self.log("using model passed as function argument")
        
        if model is None:
            self._throw("no model passed and no saved model found!!")
            
        return model

    def train(
        self,
        x_train,
        y_train=None,
        x_test=None,
        y_test=None,
        model=None,
        epochs=10,
        batch_size=32,
        force=False,
        metrics=[],
        loss=None,
        loss_weights=None,
        optimizer=None,
        verbose=1,
        use_callbacks=False,
        learning_rate=0.01,
        custom_objects={},
        compile_me=True,
        es_patience=10,
        lr_patience=9,
        lr_factor=0.5,
    ):
        callbacks = None

        w_path = self.config_file.replace(".pkl", "_weights.h5")
        if use_callbacks:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=es_patience, verbose=0),
                ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, verbose=0),
                TerminateOnNaN(),
                ModelCheckpoint(w_path, monitor='val_loss', verbose=self.VERBOSE, save_best_only=True, save_weights_only=True, mode='min')
            ]
        
        model = self.load_model(model, force, custom_objects)

        if optimizer is None:
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
            else:
                optimizer = getattr(keras.optimizers, "adam")(lr=learning_rate)
        elif isinstance(optimizer, str):
            optimizer = getattr(keras.optimizers, "Adam")(lr=learning_rate)
        if loss is None:
            if hasattr(model, "loss"):
                loss = model.loss
                loss_weights = model.loss_weights
            else:
                loss = "mse"
                loss_weights = [1.]
        
        if loss_weights is None:
            if isinstance(loss, list):
                loss_weights = [1. for i in range(len(loss))]
            else:
                loss = [loss]
                loss_weights = [1.]

        if metrics is None:
            if hasattr(model, "metrics"):
                metrics = model.metrics
            else:
                metrics = []
        if compile_me:
            model.compile(
                optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

        start = datetime.now()

        if y_train is None:
            y_train = x_train
        
        if x_test is None:
            x_test = np.zeros((0,) + x_train.shape[1:])

        if y_test is None:
            y_test = x_test.copy()


        # assert x_test.shape[0] == y_test.shape[0]
        # assert x_train.shape[0] == y_train.shape[0]
        # assert x_test.shape[1] == x_train.shape[1]
        # assert y_test.shape[1] == y_train.shape[1]


        previous_epochs = self.config['epoch_splits']

        master_epoch_n = sum(previous_epochs)
        finished_epoch_n = master_epoch_n + epochs
    
        history = odict()

        if not use_callbacks:
            try:
                for epoch in range(epochs):
                    self.log("TRAINING EPOCH {}/{}".format(master_epoch_n, finished_epoch_n))
                    nhistory = model.fit(
                        x=x_train,
                        y=y_train,
                        # steps_per_epoch=int(np.ceil(len(x_train)/batch_size)),
                        # validation_steps=int(np.ceil(len(x_test)/batch_size)),
                        validation_data=[x_test, y_test],
                        initial_epoch=master_epoch_n,
                        epochs=master_epoch_n + 1,
                        verbose=verbose,
                        callbacks=callbacks,
                        batch_size=batch_size,
                    ).history

                    if epoch == 0:
                        for metric in nhistory:
                            history[metric] = []

                    for metric in nhistory:
                        history[metric].append([master_epoch_n, nhistory[metric][0]])

                    master_epoch_n += 1

            except:
                self.error(traceback.format_exc())
                if all([len(v) == 0 for v in history]):
                    self._throw("quitting")
                self.log("saving to path " + w_path)
                model.save_weights(w_path)

            if len(list(history.values())) == 0:
                n_epochs_finished = 0
            else:
                n_epochs_finished = min(list(map(len, list(history.values()))))

        else:
            nhistory = model.fit(
                x=x_train,
                y=y_train,

                validation_data=[x_test, y_test],
                initial_epoch=master_epoch_n,
                epochs=master_epoch_n + epochs,
                verbose=verbose,
                callbacks=callbacks,
                        batch_size=batch_size,
            ).history

            for metric in nhistory:
                history[metric] = []
                for i,value in enumerate(nhistory[metric]):
                    history[metric].append([master_epoch_n + i, value])
            master_epoch_n += epochs
            n_epochs_finished = min(list(map(len, list(history.values()))))

        # self.log("EPOCH N: {}, {}".format(master_epoch_n, epochs))
        self.log("")
        self.log("trained {} epochs!".format(n_epochs_finished))
        self.log("")

        js = model.to_json()
        self.config['trained'] = True 
        self.config['model_json'] = str(js)
        # print "saving to path " + w_path
        # model.save_weights(w_path)

        # load the last model
        best = self.load_model()
        hvalues = [hv[:n_epochs_finished] for hv in list(history.values())]
        hkeys = list(history.keys())

        nmetrics = self.config['metrics'].copy()
        for key,value in zip(hkeys, hvalues):
            if key in nmetrics:
                nmetrics[key] = np.concatenate([nmetrics[key], value])
            else:
                nmetrics[key] = np.asarray(value)

        self.config['metrics'] = nmetrics

        previous_epochs.append(n_epochs_finished)
        finished_epoch_n = sum(previous_epochs)

        end = datetime.now()

        self.log("finished epoch N: {}".format(finished_epoch_n))
        # print "prev", previous_epochs


        
        self.log("model saved")


        self.config['time'] = str(end - start)
        self.config['epochs'] = epochs
        self.config['batch_size'] = self.config['batch_size'] + [batch_size,]*n_epochs_finished
        self.config['epoch_splits'] = previous_epochs
    
        return model

    def plot_metrics(
        self,
        fnmatch_criteria="*loss*",
        *args,
        **kwargs
    ):
        names = []
        metrics = []

        for mname in self.config['metrics']:
            if glob.fnmatch.fnmatch(mname, fnmatch_criteria):
                names.append(mname)
                metrics.append(np.asarray(self.config['metrics'][mname]))
                
        # break concatenated plot arrays into individual components
        plots = []
        for mn,metric in enumerate(metrics):
            splits = [0,] + list(np.where(np.diff(metric[:,0].astype(int)) > 1)[0]) + [len(metric[:,0]),]
            plots.append([])
            for i in range(len(splits[:-1])):
                plots[mn].append(metric[splits[i] + 1 : splits[i+1] + 1 , :])

        # plot em'

        fig, ax_begin, ax_end, plt_end, colors = get_plot_params(1, *args, **kwargs)

        ax = ax_begin(0)

        for color,plot,name in zip(colors[:len(plots)], plots, names):
            for subplot in plot:
                ax.plot(subplot[:,0], subplot[:,1], c=color, label=name)

        ax_end("epoch number", "metric value")
        plt_end()

    def remove(
        self,
        sure=False,
    ):
        if not sure:
            self.error("NOT DELETING: run again with keyword 'sure=True' to remove!")
        else:
            for f in [self.config_file, self.config_file.replace(".pkl", "_weights.h5")]:
                if os.path.exists(f):
                    os.remove(f)
            self.log("removed associated data files for self!")
