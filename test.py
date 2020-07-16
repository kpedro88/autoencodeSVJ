
print("Hello")

#import os
#os.system("conda info --envs")

import autoencodeSVJ.utils as utils
import autoencodeSVJ.evaluate as ev
import autoencodeSVJ.models as models
import autoencodeSVJ.trainer as trainer

import numpy as np
import tensorflow as tf
import os
import datetime
from collections import OrderedDict as odict

import keras
import keras.backend as K

import keras
import keras.backend as K

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    
def build_vae(input_dim, latent_dim, middle_arch=(100,100), loss='mse'):


    input_shape = (input_dim,)
    inputs = keras.layers.Input(shape=input_shape, name='encoder_input')
    last = inputs
    first_middle = []
    for i,n in enumerate(middle_arch):
        first_middle.append(keras.layers.Dense(n, activation='relu', name='interm_1.{}'.format(i))(last))
        last = first_middle[i]
    
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(last)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(last)
    
    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    latent_inputs = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    last = latent_inputs
    second_middle = []
    for i,n in enumerate(reversed(middle_arch)):
        second_middle.append(keras.layers.Dense(n, activation='relu', name='interm_2.{}'.format(i))(last))
        last = second_middle[i]
    outputs = keras.layers.Dense(input_dim, activation='linear', name='outputs')(last)
    
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')
    
    output = decoder(encoder(inputs)[2])
    vae = keras.models.Model(inputs, output, name='vae')
    
    reconstruction_loss = getattr(keras.losses, loss)(inputs, output)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return vae, reconstruction_loss, kl_loss

vae_custom_objects = {
    'sampling': sampling
}

vae, reco_loss, kl_loss = build_vae(19, 4)
custom_objects = vae_custom_objects.copy()

def reco_metric(y_true, y_pred):
    return K.mean(reco_loss)

def kl_metric(y_true, y_pred):
    return K.mean(kl_loss)

vae.compile(loss=[lambda x,y: K.mean(reco_loss) + 0.0001*K.mean(kl_loss)], optimizer='adam', metrics=[reco_metric, kl_metric])


sdata,sjets,sevent,sflavor = utils.load_all_data("/afs/cern.ch/work/l/llepotti/public/training_data/1500GeV_0.15/base_3/*.h5", 'SVJ')

data, jets, event, flavor = utils.load_all_data("/afs/cern.ch/work/l/llepotti/public/training_data/qcd/base_3/*.h5", 'background')
norm = data.norm(norm_type='MinMaxScaler')
train_norm, val_norm = norm.train_test_split(0.2)
snorm = data.norm(sdata, norm_type='MinMaxScaler')

history = vae.fit(batch_size=64, x=train_norm.values, y=train_norm.values, validation_data=[val_norm.values, val_norm.values], epochs=10)

