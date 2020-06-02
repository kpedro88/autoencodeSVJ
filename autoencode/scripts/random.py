import autoencodeSVJ.utils as utils
import autoencodeSVJ.evaluate as ev
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import OrderedDict as odict
import pandas as pd
import glob
import os
import tensorflow as tf
import time

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

TRAIN = True

lr = (0.0004, 0.001)
lr_factor = 0.5
es_patience = 12
target_dim = set([6, 7, 8, 9])
batch_size = set([32, 64])
norm_percentile = 25
epochs = 100

n_models_each = 50

elts = []
for dim in target_dim:
    for bs in batch_size:
        for i in range(n_models_each):
            lrv = np.random.random()*(lr[1] - lr[0]) + lr[0]
            elts.append((dim, bs, lrv))

import random
random.shuffle(elts)

for n,plist in enumerate(elts):
    dim, bs, lrv = plist
    mse = ev.ae_train(
        signal_path='data/all_signals/2000GeV_0.15/base_3/*.h5',
        qcd_path='data/background/base_3/*.h5',
        target_dim=dim,
        verbose=False,
        batch_size=bs,        
        learning_rate=lrv,
        norm_percentile=norm_percentile,
        lr_factor=lr_factor,
        es_patience=es_patience,
        epochs=epochs
    )
    print('model {} finished (mse = {:.4f})'.format(n, mse))
    print

ev.update_all_signal_evals()