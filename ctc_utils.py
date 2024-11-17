import numpy as np
import tensorflow as tf
import cv2

def convert_inputs_to_ctc_format(target_text):
    SPACE_TOKEN = '-'
    SPACE_INDEX = 4
    FIRST_INDEX = 0

    original = ' '.join(target_text.strip().lower().split(' ')).replace('.', '').replace('?', '').replace(',', '').replace("'", '').replace('!', '').replace('-', '')
    print(original)
    targets = original.replace(' ', '  ')
    targets = targets.split(' ')

    # Adding blank label
    targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

    # Transform char into index
    targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                          for x in targets])

    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from([targets])

    return train_targets, original

def ctc_loss(labels, logits, labels_length, logits_length):
    # y_true: Tr
    loss = tf.nn.ctc_loss(labels, logits, labels_length, logits_length)
    return loss

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def sparse_tensor_to_strs(sparse_tensor):
    indices= sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [ [] for i in range(dense_shape[0]) ]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def word_separator():
    return '\t'

def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def edit_distance(a,b,EOS=-1,PAD=-1):
    _a = [s for s in a if s != EOS and s != PAD]
    _b = [s for s in b if s != EOS and s != PAD]

    return levenshtein(_a,_b)

def rotate(images, angle):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0,2] += bound_w/2 - image_center[0]
    rotation_mat[1,2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    outs.append(rotated_mat)

def strech(image, factor, axis):
    height, width = image.shape[:2]
    if axis == 0:
        new_height = height
        new_width = int(width * factor)
    else:
        new_height = int(height * factor)
        new_width = width
    
    strech_mat = np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    strech_mat[axis,axis] = factor
    streched_mat = cv2.warpAffine(image, strech_mat, (new_width, new_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))


    return streched_mat
    
def scale(image, factor):
    print(image)
    height, width = image.shape[:2]

    # Calculate the new dimensions while retaining the same size
    new_height = int(height * factor)
    new_width = int(width * factor)
    
    # Resize the image using OpenCV
    resized_array = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate the padding to retain the original size
    top_pad = (height - new_height) // 2
    left_pad = (width - new_width) // 2

    if factor < 1.0:
        output_array = np.full_like(image, 255)
        output_array[top_pad:top_pad + new_height, left_pad:left_pad + new_width] = resized_array
    else:
        output_array = resized_array
        
    return output_array

def translate(image, factor, axis):
    height, width = image.shape[:2]
    translate_mat = np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float32)
    translate_mat[axis,2] = factor
    translated_mat = cv2.warpAffine(image, translate_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return translated_mat

def blur(image, factor):
    blur_factor = np.abs(int(factor)) + 1
    blurred_mat = cv2.blur(image,(int(blur_factor),int(blur_factor)))
    return blurred_mat

# Too slow for otf augmentation
def radial_distortion(image, k1, k2, center_x=None, center_y=None):
    if center_x is None:
        center_x = image.shape[1] / 2
    if center_y is None:
        center_y = image.shape[0] / 2

    #make the distorted image the same size as the original
    distorted_points = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r2 = (x - center_x) ** 2 + (y - center_y) ** 2
            r2 = np.sqrt(r2)
            distortion = k1 * r2 + k2 * r2 ** 2

            x_distorted = int(x * distortion)
            y_distorted = int(y * distortion)

            new_x = int(x + (x_distorted))
            new_y = int(y + (y_distorted))

            if new_x >= 0 and new_x < image.shape[1] and new_y >= 0 and new_y < image.shape[0]:
                distorted_points[y][x] = image[new_y][new_x]
            else:
                distorted_points[y][x] = 255

    return np.array(distorted_points)

def contrast_shift(image, factor):
    contrast_mat = np.ones(image.shape, dtype="uint8") * int(factor)
    contrasted_mat = cv2.subtract(image, contrast_mat)
    return contrasted_mat

def brightness_shift(image, factor):
    brightness_mat = np.ones(image.shape, dtype="uint8") * int(factor)
    brightened_mat = cv2.add(image, brightness_mat)
    return brightened_mat

def sharpen(image, factor):
    sharpen_factor = np.abs(int(factor))
    sharpen_mat = np.array([[-1,-1,-1],[-1,sharpen_factor,-1],[-1,-1,-1]],dtype=np.float32)
    sharpened_mat = cv2.filter2D(image, -1, sharpen_mat)
    return sharpened_mat

def salt_pepper(image, factor,):
    salt_pepper_mat = np.zeros(image.shape, dtype="uint8")
    salt_pepper_mat = cv2.randu(salt_pepper_mat,0,255)
    salt_pepper_mat = salt_pepper_mat < factor
    salt_pepper_mat = salt_pepper_mat.astype(np.uint8)
    salt_pepper_mat = salt_pepper_mat * 255
    salt_peppered_mat = cv2.add(image, salt_pepper_mat)
    return salt_peppered_mat


#eliminate the white space around the image
def crop(image):
    #remove the 
    return image


def normalize(image):
    return (255. - image)/255.

def denormalize(image):
    return (255. - image)*255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img
