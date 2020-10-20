import module.utils as utils
import module.trainer as trainer
import module.summaryProcessor as summaryProcessor

import numpy as np
import os
from collections import OrderedDict as odict

class AutoEncoderEvaluator:
    
    def set_data_paths(self, qcd_path, signals):
        if qcd_path is None:
            if 'qcd_path' in self.d:
                qcd_path = self.d['qcd_path']
            else:
                raise AttributeError
        
        self.qcd_path = qcd_path
        
        assert isinstance(signals, (dict, odict)), 'aux_signals_dict must be dict or odict with {name: path} format'
        
        self.signals = signals
        
        for signal in self.signals:
            setattr(self, signal + '_path', self.signals[signal])
            
            (data,
             jets,
             event,
             flavor) = utils.load_all_data(getattr(self, signal + '_path'), signal,
                                           include_eflow=self.eflow, hlf_to_drop=self.hlf_to_drop,
                                           include_hlf=self.hlf
                                           )
            setattr(self, signal, data)
            setattr(self, signal + '_jets', jets)
            setattr(self, signal + '_event', event)
            setattr(self, signal + '_flavor', flavor)
    
    def set_variables_from_summary(self):
        self.hlf = self.d['hlf']
        self.eflow = self.d['eflow']
        self.eflow_base = self.d['eflow_base']
        self.hlf_to_drop = list(map(str, self.d['hlf_to_drop']))
        self.seed = self.d['seed']
        self.target_dim = self.d['target_dim']
        self.input_dim = self.d['input_dim']
        self.test_split = self.d['test_split']
        self.val_split = self.d['val_split']
        self.training_output_path = self.d['training_output_path']
        self.norm_type = self.d["norm_type"]
        self.norm_percentile = self.d["norm_percentile"]
        
        self.norm_args = {"norm_type": self.norm_type, "norm_percentile": self.norm_percentile}
    
    def find_pkl_file(self):
        # try to find training pkl file and load 'er up
        if not os.path.exists(self.training_output_path + ".pkl"):
            print((self.training_output_path + ".pkl"))
            self.filepath = utils.path_in_repo(self.training_output_path + ".pkl")
            print((self.training_output_path))
            if self.training_output_path is None:
                raise AttributeError
            else:
                if self.training_output_path.endswith(".h5"):
                    self.training_output_path.rstrip(".h5")
    
    def __init__(self, summary_path, qcd_path=None, signals={}):
        
        # Set internal variables
        self.d = summaryProcessor.load_summary(summary_path)
        self.set_variables_from_summary()
        self.set_data_paths(qcd_path=qcd_path, signals=signals)
        
        (self.qcd,
         self.qcd_jets,
         self.qcd_event,
         self.qcd_flavor) = utils.load_all_data(self.qcd_path, "qcd background",
                                                include_hlf=self.hlf, include_eflow=self.eflow,
                                                hlf_to_drop=self.hlf_to_drop
                                                )
        
        self.find_pkl_file()
        self.trainer = trainer.trainer(self.training_output_path)
        self.model = self.trainer.load_model()
        
        # Set random seed to the same value as during the training
        utils.set_random_seed(self.seed)
        
        # Split input data into training, validaiton and test samples
        self.train_and_validation_data, self.test_data = self.qcd.split_by_event(test_fraction=self.test_split,
                                                                                 random_state=self.seed,
                                                                                 n_skip=len(self.qcd_jets))
        self.train_data, self.validation_data = self.train_and_validation_data.train_test_split(self.val_split,
                                                                                                self.seed)
        
        # Normalize the input
        if self.norm_type == "Custom":
            self.data_ranges = utils.percentile_normalization_ranges(self.test_data, self.norm_percentile)
            
            self.train_data.name = "qcd training data"
            self.test_data.name = "qcd test data"
            self.validation_data.name = "qcd validation data"
            
            self.train_data_normalized = self.train_data.normalize_in_range(rng=self.data_ranges)
            self.validation_data_normalized = self.validation_data.normalize(rng=self.data_ranges)
            self.test_data_normalized = self.test_data.normalize(rng=self.data_ranges)
            
            for signal in self.signals:
                setattr(self, signal + '_norm',
                        getattr(self, signal).normalize(out_name=signal + ' norm', rng=self.data_ranges))
        else:
            print("Normalization not implemented: ", self.norm_type)
        
        # self.train_data_normalized = self.train_data.norm(out_name="qcd train norm", **self.norm_args)
        
        # Get reconstruction errors and values
        errors, recons = utils.get_recon_errors(
            [self.test_data_normalized] + [getattr(self, signal + '_norm') for signal in self.signals], self.model)
        self.qcd_err, signal_errs = errors[0], errors[1:]
        
        if self.norm_type == "Custom":
            self.qcd_recon = recons[0].inverse_normalize_in_range(rng=self.data_ranges)
            
            for err, recon, signal in zip(signal_errs, recons[1:], self.signals):
                setattr(self, signal + '_err', err)
                setattr(self, signal + '_recon', recon.inverse_normalize_in_range(rng=self.data_ranges))
        else:
            print("Normalization not implemented: ", self.norm_type)
        
        self.qcd_reps = utils.data_table(self.model.layers[1].predict(self.test_data_normalized.data), name='QCD reps')
        
        for signal in self.signals:
            setattr(self, signal + '_reps',
                    utils.data_table(self.model.layers[1].predict(getattr(self, signal + '_norm').data),
                                     name=signal + ' reps'))
        
        self.qcd_err_jets = [
            utils.data_table(self.qcd_err.loc[self.qcd_err.index % 2 == i], name=self.qcd_err.name + " jet " + str(i))
            for i in range(2)]
        
        for signal in self.signals:
            serr = getattr(self, signal + '_err')
            setattr(self, signal + '_err_jets',
                    [utils.data_table(serr.loc[serr.index % 2 == i], name=serr.name + " jet " + str(i)) for i in
                     range(2)])
        
        self.test_flavor = self.qcd_flavor.iloc[self.test_data.index]
        
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
        self.norms_dict['qcd'] = self.test_data_normalized
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
    
    def split_my_jets(self, test_vec, SVJ_vec, split_by_leading_jet, split_by_flavor, include_names=False):
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
    
    def retdict(self, this, others, ):
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
        
        SVJ_pre, qcd_pre, flag_pre = self.split_my_jets(self.test_norm, self.SVJ_norm, split_by_leading_jet,
                                                        split_by_flavor, include_names=True)
        SVJ_post, qcd_post, flag_post = self.split_my_jets(self.qcd_recon, self.SVJ_recon, split_by_leading_jet,
                                                           split_by_flavor, include_names=True)
        
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
    
    def metrics(self, show_plot=True, figname='metrics', figsize=(8, 7), figloc='upper right', *args, **kwargs):
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
            print((k + ":"))
            for subk in cuts[k]:
                print((" -", str(subk) + ":", cuts[k][subk].shape))
                s += len(cuts[k][subk])
            print((" - size:", s))
        
        print((" - og SVJ size:", len(self.SVJ) / 2))
        print((" - og test size:", len(self.test) / 2))
    
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
        
        for name, cut in list(cuts.items()):
            if 'qcd' in name.lower():
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
            for jet_n, idx in list(cut.items()):
                hname = "{}_SVJ{}".format(var, jet_n)
                hist = rt.TH1F(hname, hname, bins, *rng)
                
                data = getattr(self, name + "_event").loc[idx][var]
                rtnp.fill_hist(hist, data)
                all_data[out_name].append(np.histogram(data, bins=bins, range=rng))
                histos.append(hist)
            
            f.Write()
        
        return all_data
