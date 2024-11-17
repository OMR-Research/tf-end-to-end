import tensorflow
import tensorflow as tf
import os
import logging
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)

# Returns default model parameters
def default_model_params(img_height, vocabulary_size, batch_size=16):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = batch_size
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params


def create_ctc_crnn(params):
    # TODO Assert parameters
    input = tf.compat.v1.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')
    input_shape = tf.compat.v1.shape(input)

    # Convolutional blocks
    x = input
    width_reduction = 1
    height_reduction = 1
    for i in range(params['conv_blocks']):
        x = tf.compat.v1.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.compat.v1.layers.batch_normalization(x)
        x = leaky_relu(x)

        x = tf.compat.v1.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i])

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    # Reshape to apply RNN
    features = tf.compat.v1.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    features = tf.compat.v1.reshape(features, tf.compat.v1.stack([tf.compat.v1.cast(feature_width,'int32'), input_shape[0], tf.compat.v1.cast(feature_dim,'int32')]))  # -> [width, batch, features]

    tf.compat.v1.constant(params['img_height'],name='input_height')
    tf.compat.v1.constant(width_reduction,name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        features,
        dtype=tf.float32,
        time_major=True,
    )
    rnn_outputs = tf.compat.v1.concat(rnn_outputs, 2)

    # Output layer
    logits = tf.compat.v1.layers.dense(
        rnn_outputs,
        params['vocabulary_size'] + 1,  # BLANK
        activation=None,
    )
    tf.compat.v1.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.compat.v1.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.compat.v1.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob

def load_ctc_crnn(model_path, voc_file):
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.InteractiveSession()
    # Disable Eager execution
    tf.compat.v1.disable_eager_execution()

    # Load vocabulary
    dict_file = open(voc_file,'r')
    dict_list = dict_file.read().splitlines()
    int2word = dict()
    for word in dict_list:
        word_idx = len(int2word)
        int2word[word_idx] = word
    dict_file.close()
    
    for path in [model_path, voc_file]: assert os.path.exists(path), 'File does not exist: ' + path
    assert model_path.endswith('.meta'), 'Not a meta file: ' + model_path

     # Restore weights
    saver = tf.compat.v1.train.import_meta_graph(model_path)
    saver.restore(sess,model_path[:-5])

    graph = tf.compat.v1.get_default_graph()

    input:tensorflow.Tensor = graph.get_tensor_by_name("model_input:0")
    seq_len:tensorflow.Tensor = graph.get_tensor_by_name("seq_lengths:0")
    loss:tensorflow.Tensor = graph.get_tensor_by_name("CTCLoss:0")
    rnn_keep_prob:tensorflow.Tensor = graph.get_tensor_by_name("keep_prob:0")
    height_tensor:tensorflow.Tensor = graph.get_tensor_by_name("input_height:0")
    width_reduction_tensor:tensorflow.Tensor = graph.get_tensor_by_name("width_reduction:0")
    logits = tf.compat.v1.get_collection("logits")[0]

    # Constants that are saved inside the model itself
    width_reduction, height = sess.run([width_reduction_tensor, height_tensor])

    decoded, _ = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)

    return sess, input, seq_len, decoded, loss, rnn_keep_prob, width_reduction, height, int2word