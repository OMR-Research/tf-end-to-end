import tensorflow as tf
from tensorflow import keras
import numpy as np


class OMR_model(keras.Model):
    def __init__(self, kwargs, model_params):
        model_params['vocabulary_size'] = len(open(kwargs['voc_path'],'r').read().splitlines())

        super().__init__(self)
        self.model_params = model_params
        self.directory = kwargs['corpus_path']

        width_reduction = 1
        height_reduction = 1
        
        # Set up the CNN
        filter_n = model_params['conv_filter_n'].split(',')
        filter_n = [int(x) for x in filter_n]
        kernels = model_params['conv_filter_size'].split(',')
        kernels = [(int(x.split('x')[0]), int(x.split('x')[1])) for x in kernels]
        conv_pooling_size = model_params['conv_pooling_size'].split(',')
        conv_pooling_size = [(int(x.split('x')[0]), int(x.split('x')[1])) for x in conv_pooling_size]

        for i in range(len(filter_n)):
            # Add the convolutional layer
            self.layers.append(keras.layers.Conv2D(filters=filter_n[i],
                                         kernel_size=kernels[i],
                                         padding='same',
                                         activation=None))
            self.layers.append(keras.layers.BatchNormalization())
            self.layers.append(keras.layers.LeakyReLU(alpha=model_params['leaky_relu_alpha']))
            self.layers.append(keras.layers.MaxPooling2D(pool_size=conv_pooling_size[i],
                                               strides=conv_pooling_size[i]))
            
            width_reduction = width_reduction * conv_pooling_size[i][1]
            height_reduction = height_reduction * conv_pooling_size[i][0]

        # Reshape the output of the CNN to a 2D tensor to feed into the RNN
        input_shape=(None, int(model_params['img_height']),
            int(model_params['img_width']),
            int(model_params['img_channels']))
        
        # features = tf.compat.v1.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
        feature_dim = filter_n[-1] * (int(model_params['img_height']) / height_reduction)
        feature_width = input_shape[2] / width_reduction
        # features = tf.compat.v1.reshape(features, tf.compat.v1.stack([tf.compat.v1.cast(feature_width,'int32'), input_shape[0], tf.compat.v1.cast(feature_dim,'int32')]))  # -> [width, batch, features]

        
        self.layers.append(keras.layers.Reshape(target_shape=(feature_width, feature_dim * filter_n[-1])))
        # self.add(keras.layers.Dense(model_params['rnn_units'], activation='relu'))
        # self.add(keras.layers.Dropout(model_params['rnn_dropout']))
        # self.add(keras.layers.Reshape(target_shape=(model_params['img_height'] // height_reduction, model_params['img_width'] // width_reduction * filter_n[-1])))

        # Set up the RNN
        for i in range(int(model_params['rnn_layers'])):
            self.layers.append(keras.layers.Bidirectional(keras.layers.LSTM(int(model_params['rnn_units']),
                                                                  return_sequences=True,
                                                                  dropout=float(model_params['dropout']))))
        
        # Add the output layer
        self.layers.append(keras.layers.Dense(int(model_params['vocabulary_size']), activation='softmax'))
    
    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x