import ctc_model
import ctc_utils
import ctc_training
from primus import CTC_PriMuS

import tensorflow as tf
import ctc_predict
import config
import json
import argparse
import logging
import os

BATCH_SIZE = ctc_training.BATCH_SIZE
IMG_HEIGHT = ctc_training.IMG_HEIGHT

def test(corpus_path, test_path, train_path, voc_path, voc_type, model_path, verbose=False, log_file=None):
    if verbose:
        if not os.path.exists(log_file):
            open(log_file,'w').close()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='a'
        )
        print('Logging to ' + log_file)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    logging.info('---Testing model ' + model_path + '---')
    
    primus = CTC_PriMuS(corpus_path, test_path, train_path, voc_path, voc_type)
    sess, input, seq_len, decoded, loss, rnn_keep_prob, width_reduction, height, int2word = ctc_model.load_ctc_crnn(model_path, voc_path)
    params = ctc_model.default_model_params(IMG_HEIGHT,primus.vocabulary_size,batch_size=BATCH_SIZE)
    # # test image 
    # impath = "P:/project/datasets/CameraPrIMuS/Corpus/230006613-1_2_2/230006613-1_2_2.png"
    # semantic_string = ctc_predict.predict(impath, model_path, voc_path)
    # print(semantic_string)

    # Set up TF session and initialize variables
    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())

    val_ed, val_len, val_count = ctc_training.validate(primus, params, sess, input, seq_len, rnn_keep_prob, decoded)

    logging.info('Model ' + model_path + ' tested on ' + test_path + ' with vocabulary ' + voc_path + ' and vocabulary type ' + voc_type + '.')
    logging.info('Validation set: ' + str(val_ed) + ' errors in ' + str(val_len) + ' characters (' + str(val_count) + ' samples)')
    logging.info('Validation set: ' + str(val_ed/val_len) + ' CER')
    logging.info(str(1. * val_ed / val_count) + ' (' + str(100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')
    logging.info('---End of testing---')

if __name__ == '__main__':
    configured_defaults = json.load(open(config.CONFIG_PATH,'r'))

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('-corpus', dest='corpus', type=str, required=False, help='Path to the corpus directory', default=configured_defaults['corpus_path'])
    parser.add_argument('-save_model', dest='save_model', type=str, required=False, help='Path to save the model.', default=configured_defaults['model_path'])
    parser.add_argument('-vocabulary', dest='voc', type=str, required=False, help='Path to the vocabulary file.', default=configured_defaults['voc_path'])
    parser.add_argument('-voc_type', dest='voc_type', required=False, default=configured_defaults['voc_type'], choices=['semantic','agnostic'], help='Vocabulary type.')
    parser.add_argument('-verbose', dest='verbose', action="store_true", default=False, required=False)
    parser.add_argument('-log-file', dest='log_file', type=str, required=False, default=configured_defaults['test_results_path'], help='Path to the log file.')
    parser.add_argument('-test-set', dest='test_set', type=str, required=False, default=configured_defaults['test_set_path'], help='Path to the test set file.')
    parser.add_argument('-train-set' , dest='train_set', type=str, required=False, default=configured_defaults['train_set_path'], help='Path to the train set file.')
    parser.add_argument("-save-predictions", dest='save_predictions', action="store_true", default=False, required=False, help='Save predictions to a file.')
    args = parser.parse_args()

    test(args.corpus, args.test_set, args.train_set, args.voc, args.voc_type, args.save_model, args.verbose, args.log_file)

