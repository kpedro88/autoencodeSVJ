import module.utils as utils
import module.trainer as trainer
import module.models as models
import module.summaryProcessor as summaryProcessor

from module.dataHolder import data_holder
from module.aucGetter import auc_getter

import numpy as np
import tensorflow as tf
import os
import datetime
import time
from collections import OrderedDict as odict
import pandas as pd
import glob

eflow_base_lookup = {
    12: 3,
    13: 3,
    35: 4, 
    36: 4, 
}

class ae_evaluation:
    
    def __init__(self, summary_path, qcd_path=None, SVJ_path=None, aux_signals_dict={}, custom_objects={}):
        
        self.d = summaryProcessor.load_summary(summary_path)

        print("Creating object of ae_evaluation class")
        print("\tdata: ", self.d)

        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError

        if SVJ_path is None:
            if 'signal_path' in self.d:
                SVJ_path = self.d['signal_path']
            else:
                raise AttributeError

        assert isinstance(aux_signals_dict, (dict, odict)), 'aux_signals_dict must be dict or odict with {name: path} format'

        self.signals = {"SVJ": SVJ_path,}
        self.signals.update(aux_signals_dict)

        # set path attributes for all signals
        for signal in self.signals:
            setattr(self, signal + '_path', self.signals[signal])

        self.qcd_path = qcd_path
                
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))

        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(
            self.qcd_path, "qcd background",
            include_hlf=self.hlf, include_eflow=self.eflow, hlf_to_drop=self.hlf_to_drop
        )

        # set attributes for signals
        for signal in self.signals:
            (data,
            jets,
            event,
            flavor) = utils.load_all_data(
                getattr(self, signal + '_path'), signal,
                include_hlf=self.hlf, include_eflow=self.eflow, hlf_to_drop=self.hlf_to_drop
            )
            setattr(self, signal, data)
            setattr(self, signal + '_jets', jets)
            setattr(self, signal + '_event', event)
            setattr(self, signal + '_flavor', flavor)

        # get and set random seed for reproductions
        self.seed = self.d['seed']
        utils.set_random_seed(self.seed)

        # manually set a bunch of parameters from the summary dict
        self.target_dim = self.d['target_dim']
        self.input_dim = len(self.SVJ.columns)
        self.test_split = self.d['test_split']
        self.val_split = self.d['val_split']
        self.filename = self.d['filename']
        self.filepath = self.d['filepath']

        # try to find training pkl file and load 'er up 
        if not os.path.exists(self.filepath + ".pkl"):
            print((self.filepath + ".pkl"))
            self.filepath = utils.path_in_repo(self.filepath + ".pkl")
            print((self.filepath))
            if self.filepath is None:
                raise AttributeError
            else:
                if self.filepath.endswith(".h5"):
                    self.filepath.rstrip(".h5")

        # normalization args
        self.norm_args = {}
        
        if "norm_type" in self.d:
            self.norm_args["norm_type"] = str(self.d["norm_type"])
        else:
            self.norm_args["norm_type"] = "RobustScaler"
        

        self.all_train, self.test = self.qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=len(self.qcd_jets))
        # self.all_train, self.test = self.qcd.train_test_split(self.test_split, self.seed)
        self.train, self.val = self.all_train.train_test_split(self.val_split, self.seed)

        self.train_norm = self.train.norm(out_name="qcd train norm", **self.norm_args)
        self.val_norm = self.train.norm(self.val, out_name="qcd val norm", **self.norm_args)

        self.test_norm = self.test.norm(out_name="qcd test norm", **self.norm_args)

        self.rng = utils.percentile_normalization_ranges(self.test, 25)

        # set signal norms
        for signal in self.signals:
            # self.SVJ_norm = self.SVJ.norm(out_name="SVJ norm", **self.norm_args)
            setattr(self, signal + '_norm', self.test.norm(getattr(self, signal), out_name=signal + ' norm', **self.norm_args))

        self.train.name = "qcd training data"
        self.test.name = "qcd test data"
        self.val.name = "qcd validation data"
        
        self.custom_objects = custom_objects
    
        self.instance = trainer.trainer(self.filepath)

        self.ae = self.instance.load_model(custom_objects=self.custom_objects)
        
        errors, recons = utils.get_recon_errors([self.test_norm] + [getattr(self, signal + '_norm') for signal in self.signals], self.ae)
        # [self.qcd_err, self.SVJ_err], [self.qcd_recon, self.SVJ_recon] = utils.get_recon_errors([self.test_norm, self.SVJ_norm], self.ae)
        self.qcd_err, signal_errs = errors[0], errors[1:]
        self.qcd_recon, signal_recons = self.test.inorm(recons[0], **self.norm_args), recons[1:]

        for err,recon,signal in zip(signal_errs, signal_recons, self.signals):
            setattr(self, signal + '_err', err)
            setattr(self, signal + '_recon', self.test.inorm(recon, **self.norm_args))

        self.qcd_reps = utils.data_table(self.ae.layers[1].predict(self.test_norm.data), name='QCD reps')

        for signal in self.signals:
            setattr(self, signal + '_reps', utils.data_table(self.ae.layers[1].predict(getattr(self, signal + '_norm').data), name=signal + ' reps'))

        self.qcd_err_jets = [utils.data_table(self.qcd_err.loc[self.qcd_err.index % 2 == i], name=self.qcd_err.name + " jet " + str(i)) for i in range(2)]

        for signal in self.signals:
            serr = getattr(self, signal + '_err')
            setattr(self, signal + '_err_jets', [utils.data_table(serr.loc[serr.index % 2 == i], name=serr.name + " jet " + str(i)) for i in range(2)])

        self.test_flavor = self.qcd_flavor.iloc[self.test.index]

        # all 'big lists' for signals
        names = list(self.signals.keys())
        self.dists_dict = odict([(name, getattr(self, name)) for name in names])
        self.norms_dict = odict([(name, getattr(self, name + '_norm')) for name in names])
        self.errs_dict = odict([(name, getattr(self, name + '_err')) for name in names])
        self.reps_dict = odict([(name, getattr(self, name + '_reps')) for name in names])
        self.recons_dict = odict([(name, getattr(self, name + '_recon')) for name in names])
        self.errs_jet_dict = odict([(name, getattr(self, name + '_err_jets')) for name in names])
        self.flavors_dict = odict([(name, getattr(self, name + '_flavor')) for name in names])
        

        # add qcd manually
        self.dists_dict['qcd'] = self.qcd
        self.norms_dict['qcd'] = self.test_norm
        self.errs_dict['qcd'] = self.qcd_err
        self.reps_dict['qcd'] = self.qcd_reps
        self.recons_dict['qcd'] = self.qcd_recon
        self.errs_jet_dict['qcd'] = self.qcd_err_jets
        self.flavors_dict['qcd'] = self.test_flavor

        self.all_names = list(self.norms_dict.keys())
        self.dists = list(self.dists_dict.values())
        self.norms = list(self.norms_dict.values())
        self.errs = list(self.errs_dict.values())
        self.reps = list(self.reps_dict.values())
        self.recons = list(self.recons_dict.values())
        self.errs_jet = list(self.errs_jet_dict.values())
        self.flavors = list(self.flavors_dict.values())

    def split_my_jets(self, test_vec, SVJ_vec,  split_by_leading_jet, split_by_flavor, include_names=False):
        if split_by_flavor and split_by_leading_jet:
            raise AttributeError

        SVJ_out = SVJ_vec
        qcd_out = [test_vec]
        flag = 0

        if split_by_flavor:
            qcd_out = utils.jet_flavor_split(test_vec, self.test_flavor)
            if include_names:
                for i in range(len(qcd_out)):
                    qcd_out[i].name = qcd_out[i].name + ", " + test_vec.name 

        if split_by_leading_jet:
            j1s, j2s = list(map(utils.data_table, [SVJ_vec.iloc[0::2], SVJ_vec.iloc[1::2]]))
            j1s.name = 'leading SVJ jet'
            j2s.name = 'subleading SVJ jet'
            if include_names:
                j1s.name += ", " + SVJ_vec.name
                j2s.name += ", " + SVJ_vec.name
            SVJ_out = [j1s, j2s]
            qcd_out = test_vec
            flag = 1

        return SVJ_out, qcd_out, flag

    def retdict(self, this, others,):
        ret = {}
        for elt in [this] + others:
            assert elt.name not in ret
            ret[elt.name] = elt
        return ret

    def recon(
        self,
        show_plot=True,
        SVJ=True,
        qcd=True,
        pre=True,
        post=True,        
        alpha=1,
        normed=1,
        figname='variable reconstructions',
        figsize=15,
        cols=4,
        split_by_leading_jet=False,
        split_by_flavor=False,
        *args,
        **kwargs
    ):
        assert SVJ or qcd, "must select one of either 'SVJ' or 'qcd' distributions to show"
        assert pre or post, "must select one of either 'pre' or 'post' distributions to show"
        
        this_arr, SVJ_arr = [], []

        SVJ_pre, qcd_pre, flag_pre = self.split_my_jets(self.test_norm, self.SVJ_norm, split_by_leading_jet, split_by_flavor, include_names=True)
        SVJ_post, qcd_post, flag_post = self.split_my_jets(self.qcd_recon, self.SVJ_recon, split_by_leading_jet, split_by_flavor, include_names=True)

        to_plot = []

        if SVJ:
            if pre:
                if flag_pre:
                    to_plot += SVJ_pre
                else:
                    to_plot.append(SVJ_pre)
            if post:
                if flag_post:
                    to_plot += SVJ_post
                else:
                    to_plot.append(SVJ_post)

        if qcd:
            if pre:
                if flag_pre:
                    to_plot.append(qcd_pre)
                else:
                    to_plot += qcd_pre
            if post:
                if flag_post:
                    to_plot.append(qcd_post)            
                else:
                    to_plot += qcd_post

        assert len(to_plot) > 0

        if show_plot:
            to_plot[0].plot(
                to_plot[1:],
                alpha=alpha, normed=normed,
                figname=figname, figsize=figsize,
                cols=cols, *args, **kwargs
            )
            return

        return self.retdict(to_plot[0], to_plot[1:])

    def node_reps(
        self,
        show_plot=True,
        alpha=1,
        normed=1,
        figname='node reps',
        figsize=10,
        figloc='upper right',
        cols=4,
        split_by_leading_jet=False,
        split_by_flavor=False,
        *args,
        **kwargs
    ):
         
        sig, qcd, flag = self.split_my_jets(self.qcd_reps, self.SVJ_reps, split_by_leading_jet, split_by_flavor)
        
        if flag:
            this, others = qcd, sig
        else:
            this, others = sig, qcd

        if show_plot:             
            this.plot(
                others, alpha=alpha,
                normed=normed, figname=figname, figsize=figsize,
                figloc=figloc, cols=cols, *args, **kwargs
            )
            return 

        return self.retdict(this, others)
        
    def metrics(
        self,
        show_plot=True,
        figname='metrics',
        figsize=(8,7),
        figloc='upper right',
        *args,
        **kwargs
    ):
        if show_plot:
            self.instance.plot_metrics(figname=figname, figsize=figsize, figloc=figloc, *args, **kwargs)
        return self.instance.config['metrics']
    
    def error(
        self,
        show_plot=True,
        figsize=15, normed='n', 
        figname='error for eflow variables', 
        yscale='linear', rng=((0, 0.08), (0, 0.3)), 
        split_by_leading_jet=False, split_by_flavor=False,
        figloc="upper right", *args, **kwargs
    ):
        sig, qcd, flag = self.split_my_jets(self.qcd_err, self.SVJ_err, split_by_leading_jet, split_by_flavor)

        if flag:
            this, others = qcd, sig
        else:
            this, others = sig, qcd

        if show_plot:
            this.plot(
                others, figsize=figsize, normed=normed, 
                figname=figname, 
                yscale=yscale, rng=rng, 
                figloc=figloc, *args, **kwargs
            )
            return
        return self.retdict(this, others)
    
    def roc(self, show_plot=True, metrics=['mae', 'mse'], figsize=8, figloc=(0.3, 0.2), *args, **kwargs):
        
        qcd = self.errs_dict['qcd']
        others = [self.errs_dict[n] for n in self.all_names if n != 'qcd']
        
        if show_plot:
            utils.roc_auc_plot(qcd, others, metrics=metrics, figsize=figsize, figloc=figloc, *args, **kwargs)
            return

        return utils.roc_auc_dict(qcd, others, metrics=metrics)

    def cut_at_threshold(self, threshold, metric="mae"):
        sig = utils.event_error_tags(self.SVJ_err_jets, threshold, "SVJ", metric)
        qcd = utils.event_error_tags(self.qcd_err_jets, threshold, "qcd", metric)
        return {"SVJ": sig, "qcd": qcd}

    def check_cuts(self, cuts):
        for k in cuts:
            s = 0
            print((k +":"))
            for subk in cuts[k]:
                print((" -", str(subk) + ":", cuts[k][subk].shape))
                s += len(cuts[k][subk])
            print((" - size:", s))

        print((" - og SVJ size:", len(self.SVJ)/2))
        print((" - og test size:", len(self.test)/2))

    def fill_cuts(
        self,
        cuts,
        output_dir=None,
        rng=(0., 3000.),
        bins=50,
        var="MT"
    ):
        import ROOT as rt
        import root_numpy as rtnp

        if output_dir is None:
            output_dir = os.path.abspath(".")

        all_data = {}

        for name,cut in list(cuts.items()):
            if  'qcd' in name.lower():
                oname = "QCD.root"
            elif 'SVJ' in name.lower():
                oname = "SVJ_2000_0p3.root"
            
            out_name = os.path.join(output_dir, oname)
            
            if os.path.exists(out_name):
                raise AttributeError
            print(("saving root file at " + out_name))
            f = rt.TFile(out_name, "RECREATE")
            histos = []
            all_data[out_name] = []
            for jet_n, idx  in list(cut.items()):
                hname = "{}_SVJ{}".format(var, jet_n)
                hist = rt.TH1F(hname, hname, bins, *rng)
                
                data = getattr(self, name + "_event").loc[idx][var]
                rtnp.fill_hist(hist, data)
                all_data[out_name].append(np.histogram(data, bins=bins, range=rng))
                histos.append(hist)

            f.Write()
            
        return all_data

def ae_train(
    signal_path,
    qcd_path,
    target_dim,
    output_data_path,
    hlf=True,
    eflow=True,
    seed=None,
    test_split=0.15, 
    val_split=0.15,
    train_me=True,
    batch_size=64,
    loss='mse',
    optimizer='adam',
    epochs=100,
    learning_rate=0.0005,
    custom_objects={},
    interm_architecture=(30,30),
    verbose=1, 
    hlf_to_drop=['Energy', 'Flavor'],
    norm_percentile=1,
    es_patience=10,
    lr_patience=9,
    lr_factor=0.5
):

    """Training function for basic autoencoder (inputs == outputs). 
    Will create and save a summary file for this training run, with relevant
    training details etc.

    Not super flexible, but gives a good idea of how good your standard AE is.
    """

    start_timestamp = time.time()

    if seed is None:
        seed = np.random.randint(0, 99999999)
    utils.set_random_seed(seed)

    # get all our data
    (signal,
     signal_jets,
     signal_event,
     signal_flavor) = utils.load_all_data(
        signal_path,
        "signal", include_hlf=hlf, include_eflow=eflow,
        hlf_to_drop=hlf_to_drop,
    )

    (qcd,
     qcd_jets,
     qcd_event,
     qcd_flavor) = utils.load_all_data(
        qcd_path, 
        "qcd background", include_hlf=hlf, include_eflow=eflow,
        hlf_to_drop=hlf_to_drop,
    )

    if eflow:
        qcd_eflow = len([x for x in qcd.columns if "eflow" in x])
        signal_eflow = len([x for x in signal.columns if "eflow" in x])

        assert qcd_eflow == signal_eflow, 'signal and qcd eflow basis must be the same!!'
        eflow_base = eflow_base_lookup[qcd_eflow]
    else:
        eflow_base = 0

    filename = "{}{}{}_".format('hlf_' if hlf else '', 'eflow{}_'.format(eflow_base) if eflow else '', target_dim)
    
    last_version = summaryProcessor.get_last_summary_file_version(output_data_path, filename)
    filename += "v{}".format(last_version+1)
    print(("training under filename '{}'".format(filename)))

    assert len(summaryProcessor.summary_match(filename, 0)) == 0, "filename '{}' exists already! Change version id, or leave blank.".format(filename)

    filepath = os.path.join(output_data_path, "trainingRuns", filename)
    input_dim = len(signal.columns)

    data_args = {
        'target_dim': target_dim,
        'input_dim': input_dim,
        'test_split': test_split,
        'val_split': val_split,
        'hlf': hlf, 
        'eflow': eflow,
        'eflow_base': eflow_base,
        'seed': seed,
        'filename': filename,
        'filepath': filepath,
        'qcd_path': qcd_path,
        'signal_path': signal_path,
        'arch': (input_dim,) + interm_architecture + (target_dim,) + tuple(reversed(interm_architecture)) + (input_dim,),
        'hlf_to_drop': tuple(hlf_to_drop)
    }

    all_train, test = qcd.split_by_event(test_fraction=test_split, random_state=seed, n_skip=len(qcd_jets))
    train, val = all_train.train_test_split(val_split, seed)
    
    rng = utils.percentile_normalization_ranges(train, norm_percentile)
    
    train_norm = train.norm(out_name="qcd train norm", rng=rng)
    val_norm = val.norm(out_name="qcd val norm", rng=rng)
    
    test_norm = test.norm(out_name="qcd test norm", rng=rng)
    signal_norm = signal.norm(out_name="signal norm", rng=rng)

    norm_args = {
        'norm_percentile': norm_percentile,
        'range': rng.tolist()
    }

    train.name = "qcd training data"
    test.name = "qcd test data"
    val.name = "qcd validation data"

    instance = trainer.trainer(filepath, verbose=verbose)

    aes = models.base_autoencoder()
    aes.add(input_dim)
    for elt in interm_architecture:
        aes.add(elt, activation='relu')
    aes.add(target_dim, activation='relu')
    for elt in reversed(interm_architecture):
        aes.add(elt, activation='relu')
    aes.add(input_dim, activation='linear')


    ae = aes.build()
    if verbose:
        ae.summary()

    start_time = str(datetime.datetime.now())

    train_args = {
        'batch_size': batch_size, 
        'loss': loss,
        'optimizer': optimizer,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'es_patience': es_patience,
        'lr_patience': lr_patience,
        'lr_factor': lr_factor
    }

    if verbose:
        print("TRAINING WITH PARAMS >>>")
        for arg in train_args:
            print((arg, ":", train_args[arg]))

    if train_me:
        print("Training the model")
        print("Number of training samples: ", len(train_norm.data))
        print("Number of validation samples: ", len(val_norm.data))
        
        ae = instance.train(
            x_train=train_norm.data,
            x_test=val_norm.data,
            y_train=train_norm.data,
            y_test=val_norm.data,
            model=ae,
            force=True,
            use_callbacks=True,
            custom_objects=custom_objects, 
            verbose=int(verbose),
            **train_args
        )
    else:
        print("Using existing model")
        ae = instance.load_model(custom_objects=custom_objects)

    end_time = str(datetime.datetime.now())

    [data_err, signal_err], [data_recon, signal_recon] = utils.get_recon_errors([test_norm, signal_norm], ae)
    roc_dict = list(utils.roc_auc_dict(data_err, signal_err, metrics=['mae', 'mse']).values())[0]

    total_loss = data_err[loss].mean()

    result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])
    
    vid = summaryProcessor.summary_vid(path=(output_data_path+"/summary"))
    
    time_args = {'start_time': start_time, 'end_time': end_time, 'VID': vid, 'total_loss': total_loss}
    
    summaryProcessor.dump_summary_json(result_args, train_args, data_args, norm_args, time_args,
                                       output_path = (output_data_path+"/summary"))

    # roc as figure of merit

    end_time = time.time()
    training_time = end_time - start_timestamp
    print("Training executed in: ", training_time, " s")
    
    return total_loss, ae, test_norm

def update_all_signal_evals(summary_path, signal_path, qcd_path, path='autoencode/data/aucs', update_date=None):
    """update signal auc evaluations, with path `path`.
    """
    summaries = summaryProcessor.summary(summary_path=summary_path).cfilter(
        ['*auc*', 'target_dim', 'filename', 'signal_path', 'batch*', 'learning_rate']).sort_values('mae_auc')[::-1]
    
    to_add = ['{}/{}'.format(path, f) for f in summaries.filename.values if not os.path.exists('{}/{}'.format(path, f))]
    
    to_update = []
    for f in glob.glob('{}/*'.format(path)):
        if update_date is None:
            pass
        elif datetime.datetime.fromtimestamp(os.path.getmtime(f)) < update_date:
            to_update.append(f)
    
    total = len(to_add) + len(to_update)
    print('found {} trainings total'.format(total))
    if total <= 0:
        return

    signalDict = {}

    for path in glob.glob(signal_path):
        key = path.split("/")[-3]
        signalDict[key] = path
    d = data_holder(qcd=qcd_path, **signalDict)
    d.load()
    
    print("data:", d)
    
    if len(to_add) > 0:
        print('found {} trainings to add'.format(len(to_add)))
        print('filelist to add: {}'.format('\n'.join(to_add)))
        
        
    for path in to_add:
        name = path.split('/')[-1]
        tf.compat.v1.reset_default_graph()
        a = auc_getter(filename=name, summary_path=summary_path, times=True)
        norm, err, recon = a.get_errs_recon(d)
        aucs = a.get_aucs(err)
        fmt = a.auc_metric(aucs)
        fmt.to_csv(path)
        
    if len(to_update) > 0:
        print('found {} trainings to update'.format(len(to_update)))
        print('filelist to update: {}'.format('\n'.join(to_update)))
        
        
    for path in to_update:
        name = path.split('/')[-1]
        tf.compat.v1.reset_default_graph()
        a = auc_getter(filename=name, summary_path=summary_path, times=True)
        a.update_event_range(d, percentile_n=1)
        norm, err, recon = a.get_errs_recon(d)
        aucs = a.get_aucs(err)
        fmt = a.auc_metric(aucs)
        fmt.to_csv(path)

def get_training_info_dict(filepath):
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    if not os.path.exists(filepath):
        print("Could not open file: ", filepath)
        raise AttributeError
    return trainer.pkl_file(filepath).store.copy()
    

def load_auc_table(path):
    auc_dict = {}
    for f in glob.glob('{}/*'.format(path)):
        data_elt = pd.read_csv(f)
        file_elt = str(f.split('/')[-1])
        data_elt['name'] = file_elt
        auc_dict[file_elt] = data_elt
    aucs = pd.concat(auc_dict)

    aucs['mass_nu_ratio'] = list(zip(aucs.mass, aucs.nu))

    pivoted = aucs.pivot('mass_nu_ratio', 'name', 'auc')

    return pivoted

