# Unpacking behavior for iterator-like inputs: A common pattern is to pass an iterator like object such as a tf.data.Dataset or a 
# keras_core.utils.PyDataset to fit(), which will in fact yield not only features (x) but optionally targets (y) 
# and sample weights (sample_weight). Keras requires that the output of such iterator-likes be unambiguous. The iterator should return 
# a tuple of length 1, 2, or 3, where the optional second and third elements will be used for y and sample_weight respectively.
# Any other type provided will be wrapped in a length-one tuple, effectively treating everything as x.
# When yielding dicts, they should still adhere to the top-level tuple structure, e.g. ({"x0": x0, "x1": x1}, y).
# Keras will not attempt to separate features, targets, and weights from the keys of a single dict.
# A notable unsupported data type is the namedtuple. The reason is that it behaves like both an ordered datatype (tuple) and a 
# mapping datatype (dict). So given a namedtuple of the form: namedtuple("example_tuple", ["y", "x"]) it is ambiguous whether to
# reverse the order of the elements when interpreting the value. 
#Even worse is a tuple of the form: namedtuple("other_tuple", ["x", "y", "z"]) where it is unclear if the tuple was intended to be unpacked 
# into x, y, and sample_weight or passed through as a single element to x.

import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
import numpy as np
import cv2
import pathlib

class primus(tf.data.Dataset):
    def __init__(self, kwargs, val_split=0.0):
       # each element of dataset is a tuple (image, label)



    def __new__(cls, dataset_path, img_height=128, img_width=128, batch_size=32, shuffle=True):
        dataset_path = pathlib.Path(dataset_path)
        sample_directories = list(dataset_path.glob("*"))

        instance = super(CustomDataset, cls).__new__(cls)

        instance.sample_directories = sample_directories
        instance.img_height = img_height
        instance.img_width = img_width

        # Create a tf.data.Dataset using the generator function
        instance._dataset = tf.data.Dataset.from_generator(
            generator=instance._generator,
            output_signature=(tf.TensorSpec(shape=(img_height, img_width, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
        )

        # Optionally add transformations
        if shuffle:
            instance._dataset = instance._dataset.shuffle(buffer_size=len(sample_directories))
        instance._dataset = instance._dataset.batch(batch_size)

        # Optionally add other transformations, prefetching, etc.

        return instance._dataset

# Example usage:
dataset = CustomDataset(dataset_path="path/to/dataset", img_height=128, img_width=128, batch_size=32, shuffle=True)

# Now, you can use this dataset for training your model

def read_images(self, params, fold_idx):
        images = []
        labels = []

        # This fold is already in buffer
        if self.fold_idx == fold_idx:
            return
 
        self.images = None
        self.labels = None
        
        # Read the fold into the buffer
        for sample in range(self.fold_size):
            sample_filepath = self.training_list[sample + fold_idx * self.fold_size]
            sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath

            if self.distortion_ratio > 0:
                use_distorted = ((sample + self.distortion_phase)% (int(1 / self.distortion_ratio)) == 0)
            else:
                use_distorted = False

            try:
                if use_distorted:
                    if self.use_otfa:
                        sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)
                        sample_img = ctc_otfa.apply_augmentations(sample_img, self.augmentations)
                    else:
                        sample_img = cv2.imread(sample_fullpath + '_distorted.jpg', cv2.IMREAD_GRAYSCALE)
                else:
                    sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)
                assert len(sample_img.shape) == 2
            except FileNotFoundError:
                logging.warning('Image file not found: ' + sample_fullpath + ' (skipping sample)')
                continue

            height = params['img_height']
            sample_img = ctc_utils.resize(sample_img,height)
            images.append(ctc_utils.normalize(sample_img))

            sample_full_filepath = sample_fullpath + '.' + self.voc_type
            try:
                sample_gt_file = open(sample_full_filepath, 'r')
            except FileNotFoundError:
                logging.warning('Semantic file not found: ' + sample_full_filepath + ' (skipping sample)')
                continue
            
            sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
            sample_gt_file.close()

            labels.append([self.word2int[lab] for lab in sample_gt_plain])

        self.images = images
        self.labels = labels
        self.fold_idx = fold_idx


def get_batch(self, params,fold_idx, batch_idx):
        if fold_idx > self.folds:
            raise Exception('Invalid fold index')

        self.read_images(params,fold_idx)

        batch_start_idx = batch_idx * params['batch_size']
        batch_end_idx = (batch_idx + 1) * params['batch_size']
        if batch_end_idx > self.fold_size:
            batch_end_idx = self.fold_size
        
        images = self.images[batch_start_idx:batch_end_idx]
        labels = self.labels[batch_start_idx:batch_end_idx]

        # Transform to batch
        image_widths = [img.shape[1] for img in images]
        max_image_width = max(image_widths)

        batch_images = np.ones(shape=[params['batch_size'],
                                       params['img_height'],
                                       max_image_width,
                                       params['img_channels']], dtype=np.float32)*self.PAD_COLUMN

        for i, img in enumerate(images):
            batch_images[i, 0:img.shape[0], 0:img.shape[1], 0] = img

        # LENGTH
        width_reduction = 1
        for i in range(params['conv_blocks']):
            width_reduction = width_reduction * params['conv_pooling_size'][i][1]

        lengths = [ batch_images.shape[2] / width_reduction ] * batch_images.shape[0]

        return {
            'inputs': batch_images,
            'seq_lengths': np.asarray(lengths),
            'targets': labels,
        }