import autoencodeSVJ.utils as utils
import autoencodeSVJ.trainer
import autoencodeSVJ.evaluate
import numpy as np
import tensorflow as tf
import os
import autoencodeSVJ.models
import datetime
from collections import OrderedDict as odict 
import time
import pandas as pd
import glob


eflow_base_lookup = {
    12: 3,
    13: 3,
    35: 4, 
    36: 4, 
}

class ae_evaluation:
    
    def __init__(
        self,
        name,
        qcd_path=None,
        SVJ_path=None,
        aux_signals_dict={},
        custom_objects={},
    ):
        self.name = utils.summary_by_name(name)
        self.d = utils.load_summary(self.name)

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


        self.signals = {
            "SVJ": SVJ_path,
        }

        self.signals.update(aux_signals_dict)

        # set path attributes for all signals
        for signal in self.signals:
            setattr(self, signal + '_path', self.signals[signal])

        self.qcd_path = qcd_path
                
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))

        #     SVJ_path = "data/SVJ/base_{}/*.h5".format(eflow_base)
        #     qcd_path = "data/background/base_{}/*.h5".format(eflow_base)

        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(
            self.qcd_path, 
            "qcd background", include_hlf=self.hlf, include_eflow=self.eflow,
            hlf_to_drop=self.hlf_to_drop
        )

        # set attributes for signals
        for signal in self.signals:
            (data,
            jets,
            event,
            flavor) = utils.load_all_data(
                getattr(self, signal + '_path'),
                signal, include_hlf=self.hlf, include_eflow=self.eflow,
                hlf_to_drop=self.hlf_to_drop
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
        self.norm_args = {
            "norm_type": str(self.d["norm_type"])
        }
        

        self.all_train, self.test = self.qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=len(self.qcd_jets))
        # self.all_train, self.test = self.qcd.train_test_split(self.test_split, self.seed)
        self.train, self.val = self.all_train.train_test_split(self.val_split, self.seed)

        self.train_norm = self.train.norm(out_name="qcd train norm", **self.norm_args)
        self.val_norm = self.train.norm(self.val, out_name="qcd val norm", **self.norm_args)

        self.test_norm = self.test.norm(out_name="qcd test norm", **self.norm_args)

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

    def split_my_jets(
        self,
        test_vec,
        SVJ_vec, 
        split_by_leading_jet,
        split_by_flavor,
        include_names=False,
    ):
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

    def retdict(
        self,
        this,
        others,
    ):
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
    
    def roc(
        self,
        show_plot=True,
        metrics=['mae', 'mse'],
        figsize=8,
        figloc=(0.3, 0.2),
        *args,
        **kwargs
    ):
        # split_by_leading_jet = False
        
        qcd = self.errs_dict['qcd']
        others = [self.errs_dict[n] for n in self.all_names if n != 'qcd']

        # if split_by_leading_jet:
        #     SVJ, qcd, _ = self.split_my_jets(self.qcd_err, self.SVJ_err, True, False)
        #     SVJ += [self.SVJ_err]
        #     SVJ[-1].name = "combined SVJ error"
        
        if show_plot:
            utils.roc_auc_plot(
                qcd, others,
                metrics=metrics, figsize=figsize,
                figloc=figloc, *args, **kwargs
            )
            
            return

        return utils.roc_auc_dict(
            qcd, others,
            metrics=metrics
        )

    def cut_at_threshold(
        self,
        threshold,
        metric="mae"
    ):
        sig = utils.event_error_tags(self.SVJ_err_jets, threshold, "SVJ", metric)
        qcd = utils.event_error_tags(self.qcd_err_jets, threshold, "qcd", metric)

        return {"SVJ": sig, "qcd": qcd}

    def check_cuts(
        self,
        cuts,
    ):
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
    hlf=True,
    eflow=True,
    version=None,
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
    output_data_path=None,
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

    if seed is None:
        seed = np.random.randint(0, 99999999)

    # set random seed
    utils.set_random_seed(seed)

    if output_data_path is None:
        output_data_path = os.path.join(utils.get_repo_info()['head'], "autoencode/data/training_runs")

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
    
    if version is None:
        existing_ids = [int(os.path.basename(x).rstrip('.summary').split('_')[-1].lstrip('v')) for x in utils.summary_match(filename + "v*", 0)]
        assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
        id_set = set(existing_ids)
        this_num = 0
        while this_num in id_set:
            this_num += 1
        
        version = this_num

    filename += "v{}".format(version)
    print(("training under filename '{}'".format(filename)))

    assert len(utils.summary_match(filename, 0)) == 0, "filename '{}' exists already! Change version id, or leave blank.".format(filename)

    filepath = os.path.join(output_data_path, filename)
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
        ae = instance.load_model(custom_objects=custom_objects)

    end_time = str(datetime.datetime.now())

    [data_err, signal_err], [data_recon, signal_recon] = utils.get_recon_errors([test_norm, signal_norm], ae)
    roc_dict = list(utils.roc_auc_dict(data_err, signal_err, metrics=['mae', 'mse']).values())[0]

    total_loss = data_err[loss].mean()

    result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])
    
    vid = utils.summary_vid()
    
    time_args = {'start_time': start_time, 'end_time': end_time, 'VID': vid, 'total_loss': total_loss}
    utils.dump_summary_json(result_args, train_args, data_args, norm_args, time_args)

    # roc as figure of merit
    return total_loss

    # def vae_train(
    #     signal_path,
    #     qcd_path,
    #     target_dim,
    #     hlf=True,
    #     eflow=True,
    #     version=None,
    #     seed=None,
    #     test_split=0.15, 
    #     val_split=0.15,
    #     norm_args={
    #         "norm_type": "MinMaxScaler"
    #     },
    #     train_me=True,
    #     batch_size=64,
    #     loss='mse',
    #     optimizer='adam',
    #     epochs=100,
    #     learning_rate=0.0005,
    #     interm_architecture=(30,30),
    #     output_data_path=None,
    #     verbose=1, 
    #     hlf_to_drop=['Energy', 'Flavor'],
    # ):

    #     """Training function for variational autoencoder (inputs == outputs). 
    #     Will create and save a summary file for this training run, with relevant
    #     training details etc.

    #     Not super flexible, but gives a good idea of how good your standard AE is.
    #     """

    #     if seed is None:
    #         seed = np.random.randint(0, 99999999)

    #     # set random seed
    #     utils.set_random_seed(seed)

    #     if output_data_path is None:
    #         output_data_path = os.path.join(utils.get_repo_info()['head'], "autoencode/data/training_runs")

    #     # get all our data
    #     (signal,
    #      signal_jets,
    #      signal_event,
    #      signal_flavor) = utils.load_all_data(
    #         signal_path,
    #         "signal", include_hlf=hlf, include_eflow=eflow,
    #         hlf_to_drop=hlf_to_drop,
    #     )

    #     (qcd,
    #      qcd_jets,
    #      qcd_event,
    #      qcd_flavor) = utils.load_all_data(
    #         qcd_path, 
    #         "qcd background", include_hlf=hlf, include_eflow=eflow,
    #         hlf_to_drop=hlf_to_drop,
    #     )

    #     if eflow:
    #         qcd_eflow = len(filter(lambda x: "eflow" in x, qcd.columns))
    #         signal_eflow = len(filter(lambda x: "eflow" in x, signal.columns))

    #         assert qcd_eflow == signal_eflow, 'signal and qcd eflow basis must be the same!!'
    #         eflow_base = eflow_base_lookup[qcd_eflow]
    #     else:
    #         eflow_base = 0

    #     filename = "{}{}{}_".format('hlf_' if hlf else '', 'eflow{}_'.format(eflow_base) if eflow else '', target_dim)
        
    #     if version is None:
    #         existing_ids = map(lambda x: int(os.path.basename(x).rstrip('.summary').split('_')[-1].lstrip('v')), utils.summary_match(filename + "v*", 0))
    #         assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
    #         id_set = set(existing_ids)
    #         this_num = 0
    #         while this_num in id_set:
    #             this_num += 1
            
    #         version = this_num

    #     filename += "v{}".format(version)

    #     assert len(utils.summary_match(filename, 0)) == 0, "filename '{}' exists already! Change version id, or leave blank.".format(filename)

    #     filepath = os.path.join(output_data_path, filename)
    #     input_dim = len(signal.columns)
        
    #     custom_objects = models.vae_custom_objects

    #     data_args = {
    #         'target_dim': target_dim,
    #         'input_dim': input_dim,
    #         'test_split': test_split,
    #         'val_split': val_split,
    #         'hlf': hlf, 
    #         'eflow': eflow,
    #         'eflow_base': eflow_base,
    #         'seed': seed,
    #         'filename': filename,
    #         'filepath': filepath,
    #         'qcd_path': qcd_path,
    #         'signal_path': signal_path,
    #         'arch': (input_dim,) + interm_architecture + (target_dim,) + tuple(reversed(interm_architecture)) + (input_dim,),
    #         'hlf_to_drop': tuple(hlf_to_drop),
    #         'ae_type': 'vae'
    #     }

    #     all_train, test = qcd.split_by_event(test_fraction=test_split, random_state=seed, n_skip=len(qcd_jets))
    #     train, val = all_train.train_test_split(val_split, seed)

    #     train_norm = train.norm(out_name="qcd train norm", rng=rng)
    #     val_norm = train.norm(val, out_name="qcd val norm", rng=rng)
        
    #     test_norm = test.norm(out_name="qcd test norm", rng=rng)
    #     signal_norm = test.norm(signal, out_name="signal norm", rng=rng)

    #     train.name = "qcd training data"
    #     test.name = "qcd test data"
    #     val.name = "qcd validation data"

    #     instance = trainer.trainer(filepath, verbose=verbose)
        
    #     loss_reco = loss
    #     ae, loss = models.build_vae(input_dim=input_dim, latent_dim=target_dim, middle_arch=interm_architecture, loss=loss_reco)
        
    #     ae.compile(loss=lambda x,y: loss, optimizer=optimizer)
    #     if verbose:
    #         ae.summary()

    #     start_time = str(datetime.datetime.now())

    #     train_args = {
    #         'batch_size': batch_size, 
    #         'optimizer': optimizer,
    #         'epochs': epochs,
    #         'learning_rate': learning_rate,
    #     }

    #     if verbose:
    #         print("TRAINING WITH PARAMS >>>")
    #         for arg in train_args:
    #             print(arg, ":", train_args[arg])
    
    #     if train_me:
    #         ae = instance.train(
    #             x_train=train_norm.data,
    #             x_test=val_norm.data,
    #             y_train=train_norm.data,
    #             y_test=val_norm.data,
    #             model=ae,
    #             force=True,
    #             use_callbacks=True,
    #             custom_objects=custom_objects, 
    #             verbose=int(verbose),
    #             **train_args
    #         )
    #     else:
    #         ae = instance.load_model(custom_objects=custom_objects)

    #     end_time = str(datetime.datetime.now())

    #     [data_err, signal_err], [data_recon, signal_recon] = utils.get_recon_errors([test_norm, signal_norm], ae)
    #     roc_dict = utils.roc_auc_dict(data_err, signal_err, metrics=['mae', 'mse']).values()[0]
    #     result_args = dict([(r + '_auc', roc_dict[r]['auc']) for r in roc_dict])

    #     vid = utils.summary_vid()
        
    #     time_args = {'start_time': start_time, 'end_time': end_time, 'VID': vid}
    #     utils.dump_summary_json(result_args, train_args, data_args, norm_args, time_args)

    #     # roc as figure of merit
    #     return max(result_args.values())

    # def ae_cl_train(
    #     signal_path,
    #     qcd_path,
    #     target_dim,
    #     hlf=True,
    #     eflow=True,
    #     version=None,
    #     seed=None,
    #     test_split=0.15, 
    #     val_split=0.15,
    #     norm_args={
    #         "norm_type": "MinMaxScaler"
    #     },
    #     train_me=True,
    #     batch_size=64,
    #     loss='mse',
    #     optimizer='adam',
    #     epochs=100,
    #     learning_rate=0.0005,
    #     custom_objects={},
    #     interm_architecture=(30,30),
    #     output_data_path=None,
    #     verbose=1, 
    #     hlf_to_drop=['Energy', 'Flavor'],
    # ):
        
class signal_element(object):
    def __init__(
        self,
        path,
        name
    ):
        self._hlf = None
        self._eflow = None
        self._hlf_to_drop = []
        self._name = name
        self._path = path
        self._loaded = False

    def keys(
        self
    ):
        return [elt for elt in dir(self) if not elt.startswith('__')]

    def _load(
        self,
        hlf=True,
        eflow=True,
        hlf_to_drop=['Energy', 'Flavor'],
    ):
        if all([hlf == self._hlf, self._eflow == eflow, set(self._hlf_to_drop) == set(hlf_to_drop)]) and self._loaded:
            return 

        self._hlf = hlf
        self._eflow = eflow
        self._hlf_to_drop = hlf_to_drop
        self._loaded = True
        
        (self.data,
         self.jets,
         self.event,
         self.flavor) = utils.load_all_data(
            self._path, self._name, 
            include_hlf=self._hlf, include_eflow=self._eflow, hlf_to_drop=self._hlf_to_drop
        )

    def _add_attribute(
        self,
        name,
        function,
    ):
        setattr(self, name, function(self))
        
    def _rm_attribute(
        self,
        name
    ):
        delattr(self, name)

class data_holder(object):
    def __init__(
        self,
        **kwargs
    ):
        self.KEYS = {}
        names = sorted(kwargs.keys())
        for name in names:
            path = kwargs[name]
            if name[0].isdigit():
                name = 'Zprime_' + name
            name = name.replace('.', '')
            # print('loading {} from path \'{}\'...'.format(name, path))
            setattr(self, name, signal_element(path, name))
            self.KEYS[name] = getattr(self, name)

        print(('found {} datasets'.format(len(names))))

    def load(
        self,
        hlf=True,
        eflow=True,
        hlf_to_drop=['Energy', 'Flavor']
    ):
        for k,v in list(self.KEYS.items()):
            v._load(hlf, eflow, hlf_to_drop)

    def add_attribute(
        self,
        name,
        function,
    ):
        for k,v in list(self.KEYS.items()):
            v._add_attribute(name, function)
    
    def rm_attribute(
        self,
        name
    ):
        for k,v in list(self.KEYS.items()):
            v._rm_attribute(name)

    def get(
        self,
        name
    ):
        return [getattr(v, name) for v in list(self.KEYS.values())]

class auc_getter(object):
    '''This object basically needs to be able to load a training run into memory, including all 
    training/testing fractions and random seeds. It then should take a library of signals as input
    and be able to evaluate the auc on each signal to determine a 'general auc' for all signals. 
    '''
    def __init__(
        self,
        filename,
        qcd_path=None,
        times=False
    ):
        self.times = times
        self.start()
        self.name = utils.summary_by_name(filename)
        self.d = utils.load_summary(self.name)
        
        self.norm_args = {
        }

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
        for param in ['target_dim', 'input_dim', 'test_split', 'val_split', 'filename', 'filepath']:
            setattr(self, param, self.d[param])

        if not os.path.exists(self.filepath + ".pkl"):
            print((self.filepath + ".pkl"))
            self.filepath = utils.path_in_repo(self.filepath + ".pkl")
            print((self.filepath))
            if self.filepath is None:
                raise AttributeError
            else:
                if self.filepath.endswith(".h5"):
                    self.filepath.rstrip(".h5")
                
        self.instance = trainer.trainer(self.filepath)
        self.time('init')
        
    def start(
        self,
    ):
        self.__TIME = time.time()
        
    def time(
        self,
        info=None,
    ):
        end = time.time() - self.__TIME
        if self.times:
            print((':: TIME: ' + '{}executed in {:.2f} s'.format('' if info is None else info + ' ', end)))
        return end
    
    def update_event_range(
        self,
        data,
        percentile_n,
        test_key='qcd'
    ):
        utils.set_random_seed(self.seed)
        qcd = getattr(data, test_key).data
        train, test = qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=2)
        rng = utils.percentile_normalization_ranges(train, percentile_n)
        self.norm_args['rng'] = rng
        return rng        

    def get_test_dataset(
        self,
        data,
        test_key='qcd',
    ):
        self.start()
        assert hasattr(data, test_key), 'please pass a data_holder object instance with attribute \'{}\''.format(test_key)
    
        utils.set_random_seed(self.seed)
        
        qcd = getattr(data, test_key).data
        train, test = qcd.split_by_event(test_fraction=self.test_split, random_state=self.seed, n_skip=2)


        self.time('test dataset')
        return test
    
    def get_errs_recon(
        self,
        data,
        test_key='qcd',
        **kwargs
    ):

        test = self.get_test_dataset(data, test_key)

        
        self.start()
        if 'rng' in self.norm_args:
            normed = {d: getattr(data, d).data.norm(**self.norm_args) for d in data.KEYS if d != test_key}
            normed[test_key] = test.norm(**self.norm_args)
        else:
            normed = {d: test.norm(elt.data, **self.norm_args) for d,elt in list(data.KEYS.items()) if d != test_key}
            normed[test_key] = test.norm(test, **self.norm_args)

        for key in normed:
            normed[key].name = key
        ae = self.instance.load_model()
        normed = list(normed.values())
        err, recon = utils.get_recon_errors(normed, ae, **kwargs)
        for i in range(len(err)):
            err[i].name = err[i].name.rstrip('error').strip()
        
        if 'rng' in self.norm_args:
            for i in range(len(recon)):
                recon[i] = recon[i].inorm(out_name=recon[i].name, **self.norm_args)
        else:
            for i in range(len(recon)):
                recon[i] = test.inorm(recon[i], out_name=recon[i].name, **self.norm_args)
        del ae
        self.time('recon gen')
        return [{y.name: y for y in x} for x in [normed, err, recon]]
    
    def get_aucs(
        self,
        err,
        qcd_key='qcd',
        metrics=None,
    ):
        self.start()
        derr = [v for elt,v in list(err.items()) if elt == qcd_key]
        serr = [v for elt,v in list(err.items()) if elt != qcd_key]
        
        if metrics is None:
            metrics = ['mae']

        ret = utils.roc_auc_dict(data_errs=derr, signal_errs=serr, metrics=metrics)
        self.time('auc grab')
        return ret

    def plot_aucs(
        self,
        err,
        qcd_key='qcd',
        metrics=None
    ):
        self.start()
        derr = [v for elt,v in list(err.items()) if elt == qcd_key]
        serr = [v for elt,v in list(err.items()) if elt != qcd_key]
        
        if metrics is None:
            metrics = ['mae']
        ret = utils.roc_auc_plot(data_errs=derr, signal_errs=serr, metrics=metrics)
        self.time('auc plot')
        return ret
    
    def auc_metric(
        self,
        aucs
    ):
        fmt = pd.DataFrame([(k,v['mae']['auc']) for k,v in list(aucs.items())], columns=['name', 'auc'])

        mass, nu = np.asarray([list([float(y.rstrip('GeV')) for y in x.split('_')[1:]]) for x in fmt.name]).T
        nu /= 100

        fmt['mass'] = mass
        fmt['nu'] = nu

        return fmt

def update_all_signal_evals(path='autoencode/data/aucs', update_date=None, dummy=False):
    """update signal auc evaluations, with path `path`. 
    """
    top = utils.summary().cfilter(['*auc*', 'target_dim', 'filename', 'signal_path', 'batch*', 'learning_rate']).sort_values('mae_auc')[::-1]
    eflow_base = 3
    
    print("top: ", top)
    
    to_add = ['{}/{}'.format(path, f) for f in top.filename.values if not os.path.exists('{}/{}'.format(path, f))]
    print("to add: ", to_add)
    
    to_update = []
    for f in glob.glob('{}/*'.format(path)):
        if update_date is None:
            pass
        elif datetime.datetime.fromtimestamp(os.path.getmtime(f)) < update_date:
            to_update.append(f)

    total = len(to_add) + len(to_update)
    print(('found {} trainings total'.format(total)))
    if total > 0:
        
        if dummy:
            d = None
        else:
            d = evaluate.data_holder(
                qcd='data/background/base_3/*.h5',
                **{os.path.basename(p): '{}/base_{}/*.h5'.format(p, eflow_base) for p in glob.glob('data/all_signals/*')}
            )
            d.load()
        
        if len(to_add) > 0:
            print(('found {} trainings to add'.format(len(to_add))))
            print(('filelist to add: {}'.format('\n'.join(to_add))))
        
        if not dummy:
            for path in to_add:
                name = path.split('/')[-1]
                tf.reset_default_graph()            
                a = evaluate.auc_getter(name, times=True)
                norm, err, recon = a.get_errs_recon(d)
                aucs = a.get_aucs(err)
                fmt = a.auc_metric(aucs)
                fmt.to_csv(path)
            
        if len(to_update) > 0:
            print(('found {} trainings to update'.format(len(to_update))))
            print(('filelist to update: {}'.format('\n'.join(to_update))))
            
        if not dummy:
            for path in to_update:
                name = path.split('/')[-1]
                tf.reset_default_graph()            
                a = evaluate.auc_getter(name, times=True)
                a.update_event_range(d, percentile_n=1)
                norm, err, recon = a.get_errs_recon(d)
                aucs = a.get_aucs(err)
                fmt = a.auc_metric(aucs)
                fmt.to_csv(path)

def get_training_info_dict(filepath):
    default = os.path.join(utils.get_repo_info()['head'], 'autoencode/data/training_runs/')
    fp = filepath
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    if not os.path.exists(fp):
        fp = os.path.join(default, filepath)
    if not os.path.exists(fp):
        raise AttributeError
    return trainer.pkl_file(fp).store.copy()
    
def check_training(filepath):
    info_dict = get_training_info_dict(filepath)
    idxs = set([tuple(v[:,0].tolist()) for v in list(info_dict['metrics'].values())])
    assert len(idxs) == 1, 'datafile corrupt!!'
    idx = np.asarray(idxs.pop())
    vals = {k:v[:,1] for k,v in list(info_dict['metrics'].items())}
    return pd.DataFrame(vals, index=idx)

def update_aucs(filelist):
    print(('filelist: {}'.format(filelst)))
    eflow_base = 3
    d = evaluate.data_holder(
        qcd='data/background/base_3/*.h5',
        **{os.path.basename(p): '{}/base_{}/*.h5'.format(p, eflow_base) for p in glob.glob('data/all_signals/*')}
    )
    d.load()
    
    
    for name, path in list(filelist.items()):
        tf.reset_default_graph()
        a = evaluate.auc_getter(name, times=True)
        norm, err, recon = a.get_errs_recon(d)
        aucs = a.get_aucs(err)
        fmt = a.auc_metric(aucs)
        fmt.to_csv(path)

def load_auc_table(path='autoencode/data/aucs'):
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
    # l0 = l0p.loc[:,l0p.columns.isin(set(s.filename))]
    
