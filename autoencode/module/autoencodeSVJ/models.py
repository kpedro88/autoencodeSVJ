import keras 
import keras.backend as K
from autoencodeSVJ.utils import logger
import numpy as np

def shallow(bn, n_features, central_activation='relu'):

    model = base_autoencoder()
    model.add(n_features, 'relu', )
    model.add(bn, central_activation)
    model.add(n_features, 'linear')
    
    return model.build(optimizer='adam', loss='mse')

def medium(bn, n_features, central_activation='relu'):
    model = base_autoencoder()
    model.add(n_features, 'relu')
    model.add((n_features - bn)/2 + bn)
    model.add(bn, central_activation)
    model.add((n_features - bn)/2 + bn)
    model.add(n_features, 'linear')    
    
    return model.build(optimizer='adam', loss='mse')

def deep(bn, n_features, central_activation='relu', depth=100, intermediate_layers=2):
    model = base_autoencoder()
    model.add(n_features, 'relu')
    for i in range(intermediate_layers):
        model.add(depth, 'relu')
    model.add(bn, central_activation)
    for i in range(intermediate_layers):
        model.add(depth, 'relu')
    model.add(n_features, 'linear')

    return model.build(optimizer='adam', loss='mse')

class robust_deep(logger):
    """
    quick keras implementation of robust deep autoencoder, found at
    https://dl.acm.org/citation.cfm?id=3098052
    """

    def __init__(
        self,
        base_ae,
        lambda_,
        name="robust deep autoencoder",
        verbose=True,
    ):
        if isinstance(base_ae, base_autoencoder):
            self.AE = base_ae.build()
        elif hasattr(base_ae, "compile"):
            self.AE = base_ae
        else:
            raise(AttributeError, "'base_ae' parameter must be keras model or base_autoencoder instance!")
        logger.__init__(self, VERBOSE=verbose, LOG_PREFIX="robust_deep_ae :: ")
        self.L, self.S = None, None
        self.lambda_ = lambda_

    def compile(
        self,
        *args,
        **kwargs
    ):
        return self.AE.compile(*args, **kwargs)

    def fit(
        self,
        x,
        *args,
        **kwargs
    ):
        if self.L is None:
            self.L = np.zeros(x.shape)
        if self.S is None:
            self.S = np.zeros(x.shape)

        self.log("{:.2f} % OUTLIERS".format(100.*float(len(np.where(np.linalg.norm(self.S, axis=1) > 0)[0]))/float(len(self.S))))

        self.L = x - self.S

        history = self.AE.fit(
            self.L,
            *args,
            **kwargs
        )

        self.L = self.AE.predict(self.L)
        self.S = robust_deep.l21shrink(self.lambda_, (x - self.L).T).T

        return history

    def predict(
        self,
        x
    ):  
        return self.AE.predict(x)

    def to_json(
        self,
    ):
        return self.AE.to_json()

    def save_weights(
        self,
        *args,
        **kwargs
    ):
        return self.AE.save_weights(*args, **kwargs)
    
    def load_weights(
        self,
        *args,
        **kwargs
    ):
        return self.AE.load_weights(*args, **kwargs)

    @staticmethod
    def l21shrink(
        eps,
        x,
    ):
        ret = x.copy()
        normed = np.linalg.norm(x, ord=2, axis=0)
        # loop over features
        for i in range(x.shape[1]):
            # if norm for this feature is greater than eps...
            if normed[i] > eps:
                # loop across samples
                for j in range(x.shape[0]):
                    ret[j,i] = x[j,i] - eps*x[j,i]/normed[i]
            else:
                ret[:,i] = 0.
        return ret

    @staticmethod
    def l1shrink(
        eps,
        x
    ):

        ret = np.zeros(x.shape)
        for i,elt in enumerate(x):
            if elt > eps:
                ret[i] = elt - eps
            elif elt < -eps:
                ret[i] = elt + eps
            else:
                ret[i] = 0
        return ret

class base_autoencoder(logger):

    def __init__(
        self,
        name="autoencoder",
        verbose=True,
    ):
        logger.__init__(self)
        self._LOG_PREFIX = "base_autoencoder :: "
        self.VERBOSE = verbose
        self.name = name
        self.layers = []

    def __str__(
        self,
    ):
        s = self.log('Current Structure:', True)
        for layer in self.layers:
            s += self.log("{0}: {1} nodes {2}".format(layer[0], layer[1], layer[2:]), True)
        return s

    def __repr__(
        self,
    ):
        return str(self)

    def remove(
        self,
        index=None,
    ):
        if index is None:
            index = -1
        self.layers.pop(index)

    def add(
        self,
        nodes,
        activation='relu',
        reg=None,
        name=None,
        bias_init='zeros',
        kernel_init='glorot_uniform'
    ):
        if name is None:
            name = "layer_{0}".format(len(self.layers) + 1)
        
        self.layers.append([name, nodes, activation, reg, bias_init, kernel_init])

    def build(
        self,
        encoding_index=None,
        optimizer='adam',
        loss='mse',
        metrics=['accuracy']
    ):

        assert len(self.layers) >= 3, "need to have input, bottleneck, output!"

        if encoding_index is None:
            encoding_index = self._find_bottleneck(self.layers[1:-1]) + 1

        # grab individual layers
        input_layer = self.layers[0]
        inner_interms = self.layers[1:encoding_index]
        encoded_layer = self.layers[encoding_index]
        outer_interms = self.layers[encoding_index + 1:-1]
        output_layer = self.layers[-1]

        # get necessary keras layers
        inputs = self._input(input_layer)
        encoded = self._add_layer(encoded_layer, self._add_layers(inner_interms, inputs))
        encoded_input = self._input(encoded_layer)
        outputs = self._add_layer(output_layer, self._add_layers(outer_interms, encoded_input))

        # make keras models for encoder, decoder, and autoencoder
        encoder = keras.models.Model(inputs, encoded, name='encoder')
        decoder = keras.models.Model(encoded_input, outputs, name='decoder')
        autoencoder = keras.models.Model(inputs, decoder(encoder(inputs)), name='autoencoder')

        autoencoder.compile(optimizer, loss, metrics=metrics)

        return autoencoder

    # def fit(
    #     self,
    #     x,
    #     y,
    #     validation_data,
    #     batch_size=10,
    #     *args,
    #     **kwargs
    # ):

    #     if self.autoencoder is None: 
    #         raise AttributeError, "Model not yet built!"

    #     try:
    #         self.history = self.autoencoder.fit(
    #             x=x,
    #             y=y,
    #             steps_per_epoch=int(np.ceil(len(x)/batch_size)),
    #             validation_steps=int(np.ceil(len(validation_data[0])/batch_size)),
    #             validation_data=validation_data,
    #             *args,
    #             **kwargs
    #         )

    #     except KeyboardInterrupt:
    #         return None

    #     return self.history

    # def save(
    #     self,
    #     path,
    # ):
    #     path = smartpath(path)
    #     path = smartpath(path)
    #     if not os.path.exists(path):
    #         os.mkdirs(path)
    #     to_save = ['autoencoder', 'encoder', 'decoder']
    #     filenames = [os.path.join(path, model + '.h5') for model in to_save]
    #     for filename,typename in zip(filenames, to_save):
    #         if os.path.exists(smartpath(filename)):
    #             raise AttributeError, "Model already exists at file '{}'!!".format(filename)
    #         getattr(self, typename).save(filename)

    # def load(
    #     self,
    #     path,
    # ):
    #     to_load = ['autoencoder', 'encoder', 'decoder']
    #     filenames = [os.path.join(path, model + '.h5') for model in to_load]
    #     for filename,typename in zip(filenames, to_load):
    #         if not os.path.exists(smartpath(filename)):
    #             raise AttributeError, "Model does not exist at file '{}'!!".format(filename)
    #         setattr(self, typename, keras.models.load_model(filename))

    def _find_bottleneck(
        self,
        layers,
    ):
        imin = 0
        lmin = layers[0][1]
        for i,layer in enumerate(layers):
            if layer[1] < lmin:
                imin = i
                lmin = layer[1]
        return imin

    def _add_layers(
        self,
        layers,
        base_layer,
    ):
        lnext = base_layer
        for layer in layers:
            temp = lnext
            lnext = keras.layers.Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(temp)
        return lnext

    def _add_layer(
        self,
        layer,
        base_layer,
    ):
        return keras.layers.Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(base_layer)

    def _input(
        self,
        layer,
    ):
        return keras.layers.Input(shape=(layer[1],), name=layer[0])

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
   
    vae_loss = K.mean(reconstruction_loss + kl_loss)   

    return vae, vae_loss

vae_custom_objects = {
    'sampling': sampling
}
    
# def rand_search(n):
#     def r(l):
#         return l[random.randint(0, len(l) - 1)]
#     optimizers = ['adam', 'adagrad']
#     losses = ['mean_squared_error','mean_absolute_error']
#     activations = ['linear', 'tanh']
#     bottlenecks = [4,5,6,7,8]
#     batches = [10,50,100]
#     epochs = [40]
#     pdict = odict()
    
#     import json

#     for i in range(n):

#         opt = r(optimizers)
#         loss = r(losses)
#         act = r(activations)
#         bn = r(bottlenecks)
#         batch = r(batches)
#         epoch = r(epochs)

#         strname = "{}-{}-{}-{}-{}-{}".format(bn, opt, loss, act, batch, epoch)

#         if strname in pdict:
#             continue

#         pdict[strname] = {}

#         pdict[strname]['params'] = {
#             'optimizer': opt,
#             'loss': loss,
#             'activation': act,
#             'bottleneck': bn,
#             'epochs': epoch,
#             'batchsize': batch,
#         }

#         print strname

#         pdict[strname]['out'] = basic_run(bn, opt, loss, act, epoch, batch)

#         if os.path.exists("pdict.json"):
#             old = json.load(open("pdict.json"))
#             pdict.update(old)
#         json.dump(pdict, open("pdict.json", 'w+'))

#         print pdict.keys()
