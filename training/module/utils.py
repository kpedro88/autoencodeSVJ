from module.dataTable import data_table
from module.dataLoader import data_loader
from module.trainer import trainer

from collections import OrderedDict as odict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import os
import glob
import subprocess
import chardet
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


plt.rcParams['figure.figsize'] = (10,10)
plt.rcParams.update({'font.size': 18})

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
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
#    K.set_session(sess)
