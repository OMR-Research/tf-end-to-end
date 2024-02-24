import cv2
import numpy as np
import ctc_utils
import random

class CTC_PriMuS:
    gt_element_separator = '-'
    PAD_COLUMN = 0
    validation_dict = None


    def __init__(self, corpus_dirpath, corpus_filepath, dictionary_path, semantic, distortions = False, val_split = 0.0):
        self.semantic = semantic
        self.distortions = distortions
        self.corpus_dirpath = corpus_dirpath

        # Corpus
        corpus_file = open(corpus_filepath,'r')
        corpus_list = corpus_file.read().splitlines()
        corpus_file.close()

        self.current_idx = 0

        # Dictionary
        self.word2int = {}
        self.int2word = {}
            
        dict_file = open(dictionary_path,'r')
        dict_list = dict_file.read().splitlines()
        for word in dict_list:
            if not word in self.word2int:
                word_idx = len(self.word2int)
                self.word2int[word] = word_idx
                self.int2word[word_idx] = word

        dict_file.close()

        self.vocabulary_size = len(self.word2int)
        
        
        # Train and validation split
        random.shuffle(corpus_list) 
        val_idx = int(len(corpus_list) * val_split) 
        self.training_list = corpus_list[val_idx:]
        self.validation_list = corpus_list[:val_idx]
        
        print ('Training with ' + str(len(self.training_list)) + ' and validating with ' + str(len(self.validation_list)))

    def nextBatch(self, params):
        images = []
        labels = []

        # Read files
        for _ in range(params['batch_size']):
            sample_filepath = self.training_list[self.current_idx]
            sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath

            # IMAGE
            if self.distortions:
                sample_img = cv2.imread(sample_fullpath + '_distorted.jpg', cv2.IMREAD_GRAYSCALE) # Grayscale is assumed
            else:
                sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)  # Grayscale is assumed!
            height = params['img_height']
            sample_img = ctc_utils.resize(sample_img,height)
            images.append(ctc_utils.normalize(sample_img))

            # GROUND TRUTH
            if self.semantic:
                sample_full_filepath = sample_fullpath + '.semantic'
            else:
                sample_full_filepath = sample_fullpath + '.agnostic'
            
            sample_gt_file = open(sample_full_filepath, 'r')
            sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
            sample_gt_file.close()

            labels.append([self.word2int[lab] for lab in sample_gt_plain])

            self.current_idx = (self.current_idx + 1) % len( self.training_list )


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
        
    def getValidation(self, params):
        if self.validation_dict == None:                
            images = []
            labels = []
    
            # Read files
            for sample_filepath in self.validation_list:
                sample_fullpath = self.corpus_dirpath + '/' + sample_filepath + '/' + sample_filepath
    
                # IMAGE
                sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)  # Grayscale is assumed!
                height = params['img_height']
                sample_img = ctc_utils.resize(sample_img,height)
                images.append(ctc_utils.normalize(sample_img))
    
                # GROUND TRUTH
                if self.semantic:
                    sample_full_filepath = sample_fullpath + '.semantic'
                else:
                    sample_full_filepath = sample_fullpath + '.agnostic'
                
                sample_gt_file = open(sample_full_filepath, 'r')
            
                sample_gt_plain = sample_gt_file.readline().rstrip().split(ctc_utils.word_separator())
                sample_gt_file.close()
    
                labels.append([self.word2int[lab] for lab in sample_gt_plain])
    
            # Transform to batch
            image_widths = [img.shape[1] for img in images]
            max_image_width = max(image_widths)
    
            batch_images = np.ones(shape=[len(self.validation_list),
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
    
            self.validation_dict = {
                'inputs': batch_images,
                'seq_lengths': np.asarray(lengths),
                'targets': labels,
            }
            
        
        return self.validation_dict, len(self.validation_list)
