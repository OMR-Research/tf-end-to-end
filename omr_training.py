import argparse
import logging
import json
import os
import config
import numpy as np
import cv2
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import ctc_utils
import ctc_otfa
import omr

class CTCLossWrapper(tf.Module):
    def __init__(self, label_length, logit_length, blank_index=0, ):
        self.blank_index = blank_index
        self.label_length = label_length
        self.logit_length = logit_length

    def __call__(self, logits, labels):
        ctc_loss = tf.nn.ctc_loss(
            labels=labels,
            logits=logits,
            label_length=self.label_length,
            logit_length=self.logit_length,
            logits_time_major=True,
            blank_index=self.blank_index
        )
        return tf.reduce_mean(ctc_loss)

# Convert the paths to strings
def load_and_preprocess_data(image_path, label_path, img_shape):
    image_path = image_path.numpy().decode('utf-8')
    label_path = label_path.numpy().decode('utf-8')

    # Load and preprocess the image
    image = keras.utils.load_img(image_path,
                                 color_mode='grayscale',
                                 target_size=img_shape,
                                 interpolation='bilinear',
                                 keep_aspect_ratio=False)
    
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]

    with open(label_path, 'r') as label_file:
        labels = label_file.read().strip().split('\t')

    return image, labels


def augment_data(image, labels, augments=None, aug_ratio=0.0):
    image = ctc_otfa.apply_augmentations(image, augments)
    
    return image, labels


def create_datasets(corpus_path, batch_size, padded_length, img_shape, val_split, vocabulary, augments=None, aug_ratio=0.0):
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(vocabulary)
    # encoded_vocab = tokenizer.texts_to_sequences(vocabulary)
    # pad_sequences(encoded_vocab)

    sample_directories = os.listdir(corpus_path)

    # Add the corpus path to the sample directories
    sample_directories = [corpus_path + '/' + sample_directory + '/' +sample_directory for sample_directory in sample_directories]
    np.random.shuffle(sample_directories)

    # Prepare the list of file paths
    image_paths = []
    label_paths = []
    for sample in sample_directories:
        image_paths.append(sample + '.png')
        label_paths.append(sample + '.semantic')

    # Map the file contents to the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    dataset = dataset.map(lambda image_path, label_path: tf.py_function(load_and_preprocess_data, [image_path, label_path, img_shape], [tf.float32, tf.string]))
    
    dataset = dataset.shuffle(buffer_size=1000)
    
    # Pad the labels to the same length, use the maximum length of the labels as the padding length
    # dataset = dataset.map(lambda image, labels: (image, tf.pad(labels, [[0, padded_length - tf.shape(labels)[0]]], constant_values=' ')))
    
    # Split the dataset into training and validation sets
    splitter = dataset.cardinality().numpy() - int(dataset.cardinality().numpy() * val_split)
    train_dataset = dataset.take(splitter)
    val_dataset = dataset.skip(splitter)

    # Augment the training data
    if augments is not None and kwargs['aug_ratio'] != 0.0:
        augmentation_layer = lambda image, labels: (image, labels) if np.random.rand() < (1.0 - aug_ratio) else augment_data(image, labels, augments, aug_ratio)
        train_dataset = train_dataset.map(augmentation_layer)

    # Optimize
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.cache()
    
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset
    

def train(kwargs, model_params, augments=None, aug_ratio=0.0):
    tf.config.run_functions_eagerly(True)
    logger = logging.getLogger()
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if kwargs['verbose']:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if kwargs['log_file'] is not None:
        if os.path.exists(kwargs['log_file']):
            filer_handler = logging.FileHandler(kwargs['log_file'])
            filer_handler.setFormatter(formatter)
            logger.addHandler(filer_handler)
            print('Logging to ' + kwargs['log_file'])
        else:
            logger.warning('Log file does not exist: ' + kwargs['log_file'])
    
    assert os.path.exists(os.path.dirname(kwargs['model_path'])), 'Directory does not exist: ' + os.path.dirname(kwargs['model_path'])
    for path in [kwargs['corpus_path'], kwargs['voc_path']]:
        assert os.path.exists(path), 'File does not exist: ' + path

    # Set up the datasets (training/validation sets, testing set)
    if os.path.exists(kwargs['training_dataset_path']):
        logger.info('Loading dataset from ' + kwargs['training_dataset_path'])
        train_set = tf.data.Dataset.load(kwargs['training_dataset_path'])
    else:
        assert kwargs['training_dataset_path'] != kwargs['testing_dataset_path'], 'Training and testing datasets cannot be the same.'
        for path in [kwargs['training_dataset_path'], kwargs['testing_dataset_path']]: assert not os.path.exists(path), 'File already exists: ' + path
        for path in [kwargs['training_dataset_path'], kwargs['testing_dataset_path']]: assert os.path.exists(os.path.dirname(path)), 'Directory does not exist: ' + os.path.dirname(path)

        # read the label set
        with open (kwargs['voc_path'], 'r') as voc_file:
            voc = voc_file.read().splitlines()
        voc = [' '] + voc # Add the blank label

        logger.info('Creating new datasets.')
        img_shape = (int(model_params['img_height']), int(model_params['img_width']))
        train_set, val_set = create_datasets(corpus_path=['corpus_path'],
                                             batch_size=(model_params['batch_size']),
                                             padded_length=int(model_params['padded_length']),
                                             val_split=int(kwargs['split']),
                                             img_shape=img_shape,
                                             augments=augments,
                                             aug_ratio=aug_ratio,
                                             vocabulary=voc)
        
        for image_batch, labels_batch in train_set.take(int(kwargs['sample_data'])):
            logging.debug("Labels: ", labels_batch.numpy())
            cv2.imshow('image', image_batch[0].numpy())
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        tf.data.Dataset.save(val_set, kwargs['testing_dataset_path'])
        del val_set # We don't need the validation set anymore
        del voc
        tf.data.Dataset.save(train_set, kwargs['training_dataset_path'])
        logger.info('Datasets saved to ' + kwargs['training_dataset_path'] + ' and ' + kwargs['testing_dataset_path']) 
    
    # Set up the model
    if os.path.exists(kwargs['model_path']):
        logger.info('Loading model from ' + kwargs['model_path'])
         # TODO: model loading (checkpointing)
        raise Exception('Model loading not implemented yet.')
    else:
        logger.info('Creating new model.')
        #model = omr.OMR_model(kwargs, model_params)
        model = keras.model

    my_callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=kwargs['checkpoint_path'], save_weights_only=False, verbose=1,
                                        save_best_only=True, monitor='val_loss', mode='min', save_freq='epoch'),
        keras.callbacks.TensorBoard(log_dir='./logs'),
    ]
    
    ctc_loss_wrapper = CTCLossWrapper(int(model.model_params['padded_length']),
                                      int(model.model_params['vocabulary_size']),
                                      blank_index=0)

    # ctc = lambda labels, logits: tf.py_function(func=ctc_utils.tf.nn.ctc_loss,
    #                                             inp=[labels, logits, int(model.model_params['padded_length']), int(model.model_params['vocabulary_size'])],
    #                                             tout=[tf.float32])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=ctc_loss_wrapper, metrics=[])
    model.fit(train_set, batch_size=int(model_params['batch_size']), epochs=int(model_params['max_epochs']), callbacks=my_callbacks)

    logger.info('Saving model to ' + kwargs['model_path'])
    model.save(kwargs['model_path'])

    logger.info('Training finished.')

if __name__ == '__main__':
    configured_defaults = json.load(open(config.CONFIG_PATH,'r'))

    parser = argparse.ArgumentParser(description='Train OMR Model.')
    file_group = parser.add_argument_group(title='File Paths', description='Paths to the required files.')
    file_group.add_argument('--corpus-directory', dest='corpus_path', type=str, required=False, help='Path to the corpus (PrIMus dataset directory).', default=configured_defaults['corpus_path'])
    file_group.add_argument('--model-filepath', dest='model_path', type=str, required=False, help='Path to save the model.', default=configured_defaults['model_path'])
    file_group.add_argument('--checkpoint-filepath', dest='checkpoint_path', type=str, required=False, default=configured_defaults['checkpoint_path'], help='Path to the checkpoint file.')
    file_group.add_argument('--vocabulary-filepath', dest='voc_path', type=str, required=False, help='Path to the vocabulary file.', default=configured_defaults['voc_path'])
    file_group.add_argument('--model-params-filepath', dest='model_params_path', type=str, required=False, default=configured_defaults['model_params_path'], help='Path to the model parameters file.')
    file_group.add_argument('--otfa-params-filepath', dest='otfa_params', type=str, required=False, default=configured_defaults['otfa_params'], help='Path to the OTFA parameters file.')
    file_group.add_argument('--training_dataset-filepath', dest='training_dataset_path', type=str, required=False, default=configured_defaults['training_dataset_path'], help='Tensorflow training dataset file.')
    file_group.add_argument('--testing_dataset-filepath', dest='testing_dataset_path', type=str, required=False, default=configured_defaults['testing_dataset_path'], help='Tesorflow testing dataset file.')

    training_params_group = parser.add_argument_group('Training Parameters')
    training_params_group.add_argument('-voc_type', dest='voc_type', required=False, default=configured_defaults['voc_type'], choices=['semantic','agnostic'], help='Vocabulary type.')
    training_params_group.add_argument('-validate_batches', dest='validate_batches', action="store_true", default=False, required=False, help='log the SER after each epoch.')
    training_params_group.add_argument('-split', dest='split', type=float, required=False, help='Test/Validation split.', default=float(configured_defaults['val_split']))

    out_group = parser.add_argument_group('Output Parameters')
    out_group.add_argument('-verbose', dest='verbose', action="store_true", default=False, required=False, help='Print verbose output.')
    out_group.add_argument('-log-file', dest='log_file', type=str, required=False, default=configured_defaults['log_path'], help='Path to the log file.')

    aug_group = parser.add_argument_group('Data Augmentation Parameters')
    aug_group.add_argument('-aug-ratio', dest='aug_ratio', type=float, required=False, default=0.0, help='Ratio of distorted images used for training.')
    aug_group.add_argument('-sample-data', dest='sample_data', type=int, required=False, default=0, help='Number of samples to show.')
    args = parser.parse_args()
    
    model_params = json.load(open(args.model_params_path,'r'))
    augments = ctc_otfa.read_augmentations(args.otfa_params)
    kwargs = vars(args)
    
    train(kwargs, model_params, augments, args.aug_ratio)