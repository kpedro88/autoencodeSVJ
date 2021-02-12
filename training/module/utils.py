from module.DataTable import DataTable

from collections import OrderedDict as odict
from sklearn.metrics import roc_curve, roc_auc_score

import os
import glob
import subprocess
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

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


def get_errors(true, pred, out_name="errors", functions=["mse", "mae"], names=[None, None], index=None):
    
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
    return DataTable(
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
            tagged.append(DataTable(df_to_write.iloc[idx], headers=list(df_to_write.columns), name=tag_names[region]))
        else:
            tagged.append(DataTable(df_to_write.iloc[idx].drop(column_name, axis=1), name=tag_names[region]))
    return tagged, dict([(tag_names[k], v) for k,v in list(index.items())])


def smartpath(path):
    if path.startswith("~/"):
        return path
    return os.path.abspath(path)


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
            DataTable(
                data=np.asarray(to_add),
                headers=headers[h],
                name="jet {}".format(h)
            )
        )

    full = DataTable(
        data=np.vstack([jt.df for jt in jets]),
        headers=jets[0].headers,
        name="all jet data"
    )
    return full, jets


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


def get_recon_errors(data_list, autoencoder, **kwargs):

    if not isinstance(data_list, dict):
        print("ERROR -- get_recon_errors expects a dictionary!!")
        exit(0)
    
    recon = {}
    errors = {}
    
    for key, data in data_list.items():
        recon[key] = DataTable(pd.DataFrame(autoencoder.predict(data.data), columns=data.columns, index=data.index),
                               name="{0} pred".format(data.output_file_prefix))
        
        errors[key] = get_errors(recon[key].data, data.data, out_name="{0}".format(data.output_file_prefix), index=data.df.index, **kwargs)
        
    return errors, recon


def roc_auc_dict(data_errs, signal_errs, metrics=['mse', 'mae']):
    
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
        
        ret[signal_err.output_file_prefix] = {}
        
        for j,metric in enumerate(metrics):
            ret[signal_err.output_file_prefix][metric] = {}
            pred = np.hstack([signal_err[metric].values, data_err[metric].values])
            true = np.hstack([np.ones(signal_err.shape[0]), np.zeros(data_err.shape[0])])

            roc = roc_curve(true, pred)
            auc = roc_auc_score(true, pred)
            
            ret[signal_err.output_file_prefix][metric]['roc'] = roc
            ret[signal_err.output_file_prefix][metric]['auc'] = auc

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
        
            ax.plot(roc[0], roc[1], styles[j%len(styles)], c=colors[i%len(colors)], label='{} {}, AUC {:.4f}'.format(signal_err.output_file_prefix, metric, auc))

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


def glob_in_repo(globstring):
    repo_head = get_repo_info()['head']
    files = glob.glob(os.path.abspath(globstring))
    
    if len(files) == 0:
        files = glob.glob(os.path.join(repo_head, globstring))
    
    return files





def jet_flavor_split(to_split, ref=None):
    if ref is None:
        ref = to_split
    return split_table_by_column("Flavor", ref, tag_names=delphes_jet_tags_dict, df_to_write=to_split, keep_split_column=False)[0]





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
    """ Dictionary tags """
    return dict(
        [
            (
                i,
                tags[tags.sum(axis=1) == i].index
            ) for i in range(tags.shape[1] + 1)
        ]
    )


def event_error_tags(err_jets, error_threshold, name, error_metric="mae"):
    tag = [err[error_metric] > error_threshold for err in err_jets]
    tag_idx = get_event_index(tag)
    tag_data = [d.loc[tag_idx + i] for i,d in enumerate(tag)]
    jet_tags = DataTable(
        pd.DataFrame(
            np.asarray(tag_data).T,
            columns=['jet {}'.format(i) for i in range(len(tag))],
            index=tag_idx/2,
        ),
        name=name + " jet tags",
    )
    return tagged_jet_dict(jet_tags)


def path_in_repo(filename):
    head = get_repo_info()['head']
    suffix = ""
    comps = list(filter(len, filename.split(os.path.sep)))
    for i in range(len(comps)):
        considered = os.path.join(head, os.path.join(*comps[i:])) + suffix
        if os.path.exists(considered):
            return considered
    return None


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
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
