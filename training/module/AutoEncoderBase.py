from module.logger import logger

import keras

class AutoEncoderBase(logger):

    def __init__(self, name="autoencoder", verbose=True):
        logger.__init__(self)
        self._LOG_PREFIX = "base_autoencoder :: "
        self.VERBOSE = verbose
        self.name = name
        self.layers = []

    def __str__(self):
        s = self.log('Current Structure:', True)
        for layer in self.layers:
            s += self.log("{0}: {1} nodes {2}".format(layer[0], layer[1], layer[2:]), True)
        return s

    def __repr__(self):
        return str(self)

    def add(self, nodes, activation='relu', reg=None, name=None, bias_init='zeros', kernel_init='glorot_uniform'):
        if name is None:
            name = "layer_{0}".format(len(self.layers) + 1)
        
        self.layers.append([name, nodes, activation, reg, bias_init, kernel_init])

    def build(self, encoding_index=None, optimizer='adam', loss='mse', metrics=['accuracy']):

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

    def _find_bottleneck(self, layers):
        imin = 0
        lmin = layers[0][1]
        for i,layer in enumerate(layers):
            if layer[1] < lmin:
                imin = i
                lmin = layer[1]
        return imin

    def _add_layers(self, layers, base_layer):
        lnext = base_layer
        for layer in layers:
            temp = lnext
            lnext = keras.layers.Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(temp)
        return lnext

    def _add_layer(self, layer, base_layer):
        return keras.layers.Dense(layer[1], activation=layer[2], activity_regularizer=layer[3], name=layer[0], bias_initializer=layer[4], kernel_initializer=layer[5])(base_layer)

    def _input(self, layer):
        return keras.layers.Input(shape=(layer[1],), name=layer[0])