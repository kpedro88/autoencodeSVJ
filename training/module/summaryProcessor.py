import module.utils as utils
from module.AucGetter import AucGetter
from module.DataHolder import DataHolder

import os
import json
import glob
import datetime
import pandas as pd
import tensorflow as tf

from pathlib import Path
from collections import OrderedDict


def dump_summary_json(*dicts, output_path):
    summary_dict = OrderedDict()
    
    for d in dicts:
        summary_dict.update(d)
    
    assert 'training_output_path' in summary_dict, 'NEED to include a filename arg, so we can save the dict!'
    
    fpath = os.path.join(output_path, summary_dict['training_output_path'].split("/")[-1] + '.summary')
    
    print("summary path: ", fpath)
    
    if os.path.exists(fpath):
        newpath = fpath
        
        while os.path.exists(newpath):
            newpath = fpath.replace(".summary", "_1.summary")
        
        # just a check
        assert not os.path.exists(newpath)
        fpath = newpath
    
    summary_dict['summary_path'] = fpath
    
    with open(fpath, "w+") as f:
        json.dump(summary_dict, f)
    
    return summary_dict


def summary_vid(path=""):
    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(path, "VID")
    
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            vid = int(file.read().strip('\n').strip())
            return vid
    else:
        file = open(filepath, "w")
        file.write("0\n")
        return 0


def summary_by_name(name):
    if not name.endswith(".summary"):
        name += ".summary"
    
    if os.path.exists(name):
        return name
    
    matches = summary_match(name)
    
    if len(matches) == 0:
        raise AttributeError
    elif len(matches) > 1:
        raise AttributeError
    
    return matches[0]


def load_summary(path):
    assert os.path.exists(path)
    with open(path, 'r') as f:
        summary = json.load(f)
    return summary


def summary(summary_path, defaults={'hlf_to_drop': ['Flavor', 'Energy']}):
    files = glob.glob(os.path.join(summary_path, "*.summary"))
    
    data = []
    for f in files:
        with open(f) as to_read:
            d = json.load(to_read)
            d['time'] = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            for k, v in list(defaults.items()):
                if k not in d:
                    d[k] = v
            data.append(d)
    
    if len(data)==0:
        print("WARNING - no summary files found!!")
        return None
    
    return utils.DataTable(pd.DataFrame(data), name='summary')
    

def summary_match(search_path, verbose=True):
    ret = glob.glob(search_path)
    if verbose:
        print("Summary matches search_path: ", search_path, "\tglob path: ", ret)
        print(("found {} matches with search '{}'".format(len(ret), search_path)))
    return ret


def get_last_summary_file_version(summary_path, filename):
    summary_search_path = summary_path + filename + "v*"
    summary_files = summary_match(summary_search_path, verbose=False)
    
    existing_ids = []
    
    for file in summary_files:
        version_number = os.path.basename(file).rstrip('.summary').split('_')[-1].lstrip('v')
        
        existing_ids.append(int(version_number))
    
    assert len(existing_ids) == len(set(existing_ids)), "no duplicate ids"
    id_set = set(existing_ids)
    version = 0
    while version in id_set:
        version += 1
    
    return version-1


def save_all_missing_AUCs(summary_path, signals_path, AUCs_path):
    """
    Saves values of AUCs for all signals for all summaries for which AUCs file does not exist yet.
    The QCD path will be read from the corresponding summary file.
    """
    
    signalDict = {}
    for path in glob.glob(signals_path):
        key = path.split("/")[-3]
        signalDict[key] = path
    
    summaries = summary(summary_path=summary_path)
    
    for index, row in summaries.df.iterrows():
        path = row.training_output_path
        filename = path.split("/")[-1]
        auc_path = AUCs_path + "/" + filename
        
        if not os.path.exists(auc_path):
            tf.compat.v1.reset_default_graph()
            auc_getter = AucGetter(filename=filename, summary_path=summary_path, print_times=True)

            data_holder = DataHolder(qcd=row.qcd_path, **signalDict)
            data_holder.load()
            norm, err, recon = auc_getter.get_errs_recon(data_holder)
            
            ROCs = auc_getter.get_aucs(err)
            AUCs = auc_getter.auc_metric(ROCs)
            AUCs.to_csv(auc_path)