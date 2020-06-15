NEW_PARAMS = False

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

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow

matplotlib.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))

elts_name = 'elts.npz'

lr = (0.0004, 0.001)
lr_factor = 0.5
es_patience = 12
target_dim = set([6, 7, 8, 9])
batch_size = set([32, 64])
norm_percentile = 25
epochs = 100
n_models_each = 50

if os.path.exists(elts_name) and not NEW_PARAMS:
    with open(elts_name, 'r') as f:
        elts = np.load(f)
else:
    elts = []
    for dim in target_dim:
        for bs in batch_size:
            for i in range(n_models_each):
                lrv = np.random.random()*(lr[1] - lr[0]) + lr[0]
                elts.append((dim, bs, lrv))

    import random
    random.shuffle(elts)

    with open(elts_name, 'w+') as f:
        np.save(f, elts)

if NEW_PARAMS or not os.path.exists('spot'):
    spot = '0'
    with open('spot', 'w+') as f:
        f.write(spot)
else:
    with open('spot', 'r') as f:
        spot = f.read()

for n, plist in enumerate(elts):
    if n < int(spot):
        continue
    dim, bs, lrv = plist
    reset_keras()
    print('Training with batch {}, lr {:.7f}'.format(bs, lrv))
    mse = ev.ae_train(
        signal_path='data/all_signals/2000GeV_0.15/base_3/*.h5',
        qcd_path='data/background/base_3/*.h5',
        target_dim=int(dim),
        verbose=False,
        batch_size=int(bs),        
        learning_rate=lrv,
        norm_percentile=norm_percentile,
        lr_factor=lr_factor,
        es_patience=int(es_patience),
        epochs=int(epochs)
    )
    print('model {} finished (mse = {:.4f})'.format(n, mse))
    spot = str(n + 1)
    with open('spot', 'w') as f:
        f.write(spot)


ev.update_all_signal_evals()
