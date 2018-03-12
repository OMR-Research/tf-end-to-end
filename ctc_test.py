import argparse
import tensorflow as tf
from primus import CTC_PriMuS
import ctc_model
import utils
import cv2
import numpy as np

#http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

parser = argparse.ArgumentParser(description='Evaluate test set with a trained model (CTC).')
parser.add_argument('-corpus', dest='corpus', type=str, default='/home/data/PriMuS/Corpus/', help='Path to the corpus.')
parser.add_argument('-set',  dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)


#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(args.model+'.meta')
#saver.restore(sess,tf.train.latest_checkpoint(args.model))
saver.restore(sess,args.model)

graph = tf.get_default_graph()

inputs = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)


val_ed = 0
val_err = 0
val_len = 0
val_count = 0

# Vocabulary
word2int = {}
int2word = {}

dict_file = file(args.voc_file,'r')
dict_list = dict_file.read().splitlines()
for word in dict_list:
    if not word in word2int:
        word_idx = len(word2int)
        word2int[word] = word_idx
        int2word[word_idx] = word

dict_file.close()
    
# Loop over the corpus
corpus_file = file(args.set,'r')
corpus_list = corpus_file.read().splitlines()
corpus_file.close()    

validation_size = len(corpus_list)

val_idx = 0
for batch_file in corpus_list: 
    if val_idx % 50 == 0:
        print (str(val_idx) + ' of ' + str(validation_size))
        
    sample_fullpath = args.corpus + '/' + batch_file + '/' + batch_file

    # IMAGE
    sample_img = cv2.imread(sample_fullpath + '.png', False)  # Grayscale is assumed!
    sample_img = utils.resize(sample_img,HEIGHT)
    sample_img = utils.normalize(sample_img)

    # GROUND TRUTH
    sample_gt_file = file(sample_fullpath + '.agnostic', 'r')
    sample_gt_plain = sample_gt_file.readline().rstrip().split(utils.word_separator())
    sample_gt_file.close()

    label = [word2int[lab] for lab in sample_gt_plain]
    
    batch_image = np.asarray(sample_img).reshape(1,sample_img.shape[0],sample_img.shape[1],1)
    
    # LENGTH
    length = [ batch_image.shape[2] / WIDTH_REDUCTION ]        
          
    # PREDICTION
    prediction = sess.run(decoded, {
            inputs: batch_image,
            seq_len: length,
            rnn_keep_prob: 1.0            
        })

    str_predictions = utils.sparse_tensor_to_strs(prediction)

    # EVALUATION    
    ed = utils.edit_distance(str_predictions[0], label)
    if ed != 0:
        val_err = val_err + 1    
    val_ed = val_ed + ed
    val_len = val_len + len(label)
    val_count = val_count + 1
    
    # Counter        
    val_idx = val_idx + 1

print ('Samples: ' + str(val_count))
print ('Acc Err: ' + str(val_err) + ' (Avg. Err: ' + str(1. * val_err / val_count) + ')')
print ('Acc Ed: ' + str(val_ed) + ' (Avg. Ed: ' + str(1. * val_ed / val_count) + ')')
print ('SER: ' + str(100. * val_ed / val_len))
        