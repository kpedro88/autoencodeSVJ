import module.utils as utils
from module.aucGetter import auc_getter
from module.dataHolder import data_holder

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
    
    assert 'filename' in summary_dict, 'NEED to include a filename arg, so we can save the dict!'
    
    fpath = os.path.join(output_path, summary_dict['filename'] + '.summary')
    
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
    
    print("Summary by name: ", name)
    
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
    
    return utils.data_table(pd.DataFrame(data), name='summary')
    


def summary_match(search_path, verbose=True):
    ret = glob.glob(search_path)
    if verbose:
        print("Summary matches search_path: ", search_path, "\tglob path: ", ret)
        print(("found {} matches with search '{}'".format(len(ret), search_path)))
    return ret


def summary_by_features(**kwargs):
    data = summary(include_outdated=True)
    
    for k in kwargs:
        if k in data:
            data = data[data[k] == kwargs[k]]
    
    return data


def get_last_summary_file_version(output_path, filename):
    summary_search_path = output_path + "/summary/" + filename + "v*"
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


def save_all_missing_AUCs(summary_path, signals_path, qcd_path, AUCs_path):
    """
    Saves values of AUCs for all signals for all summaries for which AUCs file does not exist yet
    """
    
    signalDict = {}
    for path in glob.glob(signals_path):
        key = path.split("/")[-3]
        signalDict[key] = path
    d = data_holder(qcd=qcd_path, **signalDict)
    d.load()
    
    for filename in summary(summary_path=summary_path).filename.values:
        auc_path = AUCs_path + "/" + filename
        
        print("path: ", auc_path)
        if not os.path.exists(auc_path):
            print("\t adding")
            tf.compat.v1.reset_default_graph()
            a = auc_getter(filename=filename, summary_path=summary_path, times=True)
            norm, err, recon = a.get_errs_recon(d)
            aucs = a.get_aucs(err)
            fmt = a.auc_metric(aucs)
            fmt.to_csv(auc_path)