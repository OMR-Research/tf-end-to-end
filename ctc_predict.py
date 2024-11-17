import argparse
import tensorflow as tf
import ctc_utils
import ctc_model
import cv2
import numpy as np

# Predict the semantic string of a music score image
def predict(image_path, model_path, voc_file):
    sess = tf.compat.v1.InteractiveSession()
    sess, input, seq_len, decoded, loss, rnn_keep_prob, width_reduction, height, int2word = ctc_model.load_ctc_crnn(model_path, voc_file)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = ctc_utils.resize(image, height)
    image = ctc_utils.normalize(image)
    image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
    assert image.shape[1] == height
    assert image.shape[3] == 1

    seq_lengths = [ image.shape[2] / width_reduction ]
    
    prediction = sess.run(decoded,
                        feed_dict={
                            input: image,
                            seq_len: seq_lengths,
                            rnn_keep_prob: 1.0,
                        })

    str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
    semantic_string = ''
    for w in str_predictions[0]:
        semantic_string += (int2word[w]) +'\t'
    semantic_string = semantic_string[:-1]
    
    return semantic_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
    parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
    args = parser.parse_args()
    
    print(predict(args.image, args.model, args.voc_file))