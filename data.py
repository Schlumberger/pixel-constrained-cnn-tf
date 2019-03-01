from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 


def im_preprocess(im_shape, num_channels):
    '''
    Returns a map function to decode a fixed shape image (defined in 'im_shape') from a filename. 

    Params:
    im_shape: A list of 2 elements: [new_height, new_width]. The new fixed shape for the images. 

    '''
    def preprocess(filename):
        '''
        Returns a fixed shape image 

        Params:
        filename: Location of the image  
        '''
        im_string = tf.read_file(filename)
        im_decoded = tf.image.decode_image(im_string, channels = num_channels)
        im_resized = tf.image.resize_image_with_crop_or_pad(im_decoded, im_shape[0], im_shape[1])
        im_resized.set_shape([im_shape[0],im_shape[1],num_channels])
        return im_resized

    return preprocess

def resize_func(im_shape, num_channels):

    def masked_resize(masked_image):
        resized_masked_image = tf.concat([masked_image[0],tf.expand_dims(masked_image[1][:,:,0], axis = 2)], axis = 2)
        resized_masked_image.set_shape([im_shape[0],im_shape[1],num_channels+1])
        return resized_masked_image

    return masked_resize
    
def single_random_mask(im_shape, num_visible):
    '''
    Returns random mask where 0 corresponds to a hidden value and 1 to a visible value.
    Shape of mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    num_visible: Number of visible values (int). 

    Code credit: Emilien Dupont

    '''
    # Get random measurements
    height, width = im_shape 
    measurements = np.random.choice(range(height * width), size = num_visible, replace = False)

    # Create empty mask
    mask = np.zeros(im_shape)

    # Update mask with measurements
    for m in measurements:
        row = int(m/width)
        col = m % width 
        mask[row,col] = 1

    return mask 

def bottom_mask(im_shape, num_rows):
    '''
    Masks all the output except the bottom |num_rows| rows. 
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    num_rows: Number of bottom rows to be visible (int).

    '''
    mask = np.zeros(im_shape)
    mask[-num_rows:,:] = 1

    return mask 

def random_bottom_mask(image, min_val = 1, max_val = 8):
    '''
    Masks all the output except the bottom |num_rows| rows, where |num_rows| is randomly generated
    Shape of the mask is same as img_size.
    
    Params:
    image: Image that has to be masked. Shape is [height, width, channels]
    min_val: Minimum number of bottom rows to be visible
    max_val: Maximum number of bottom rows to be visible 
    '''
    im_shape = image.shape 

    mask = np.zeros(im_shape)
    num_rows = np.random.randint(min_val, max_val)
    mask[-num_rows:,:,0] = 1

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask 

def edge_mask(im_shape, num_pixels):
    '''
    Masks all the output except the num_pixels thick edge of the image.
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]

    '''
    mask = np.zeros(im_shape)
    mask[:num_pixels,:] = 1
    mask[-num_pixels:,:] = 1
    mask[:,:num_pixels] = 1
    mask[:,-num_pixels:] = 1

    return mask 

def random_edge_mask(image, min_val = 1, max_val = 4):
    '''
    Masks all the output except the num_pixels thick edge of the image.
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    min_val: Minimum number of edge pixels to be visible
    max_val: Maximum number of edge pixels to be visible
    '''
    im_shape = image.shape 

    mask = np.zeros(im_shape)
    num_pixels = np.random.randint(min_val, max_val)
    mask[:,:,0] = edge_mask(im_shape[:2], num_pixels)

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask 

def center_mask(im_shape, num_pixels):
    '''
    Masks all the output except the num_pixels by num_pixels central square
    of the image.
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    '''
    mask = np.zeros(im_shape)
    height = im_shape[0]
    width = im_shape[1]
    lower_height = int(height / 2 - num_pixels / 2)
    upper_height = int(height / 2 + num_pixels / 2)
    lower_width = int(width / 2 - num_pixels / 2)
    upper_width = int(width / 2 + num_pixels / 2)
    mask[lower_height:upper_height, lower_width:upper_width] = 1

    return mask 

def random_center_mask(image, min_val = 1, max_val = 4):
    '''
    Masks all the output except the num_pixels by num_pixels central square
    of the image.
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    min_val: Minimum number of central square length to be visible
    max_val: Maximum number of central square length to be visible
    '''
    im_shape = image.shape 

    mask = np.zeros(im_shape)
    num_pixels = np.random.randint(min_val, max_val)
    mask[:,:,0] = center_mask(im_shape[:2], num_pixels)

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask

def rectangular_mask(im_shape, height, width):
    '''
    Returns a mask with a rectangle of the spcified height and width of visible pixels. 
    Position of the rectange is chosen randomly. 
    Shape of the mask is same as img_size.

    Params:
    im_shape: A list of 2 elements: [height, width]
    '''
    mask = np.zeros(im_shape)
    img_height = im_shape[0]
    img_width = im_shape[1]

    # Sample top left corner of unmasked rectangle
    top_left = np.random.randint(0, img_height - 1), np.random.randint(0, img_width - 1)
    rect_height = min(height, img_height - top_left[0])
    rect_width = min(width, img_width - top_left[1])
    bottom_right = top_left[0] + rect_height, top_left[1] + rect_width
    mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1

    return mask 

def random_rectangular_mask(image, max_height, max_width):
    '''
    Returns a mask with a random rectangle of visible pixels.
    '''
    im_shape = image.shape 

    mask = np.zeros(im_shape)
   
    rect_height = np.random.randint(1,max_height)
    rect_width = np.random.randint(1,max_width)
    mask[:,:,0] = rectangular_mask(im_shape[:2], rect_height, rect_width)

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask

def blob_mask(im_shape, num_iter, threshold):
    '''
    Generates a blob mask

    Params:
    num_iter: Number of iterations to be used for each blob.
    threshold: Threshold used to either hide or make pixel visible. 
    '''
    mask = np.zeros(im_shape)
    img_height = im_shape[0]
    img_width = im_shape[1]

    # Defines the shifts around the central pixel which may be unmasked
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Sample random initial position
    init_pos = np.random.randint(0, img_height - 1), np.random.randint(0, img_width - 1)
    mask[init_pos[0],init_pos[1]] = 1

    # Initialize the list of seed positions 
    seed_positions = [init_pos]

    # Randomly expand blob
    for i in range(num_iter):
        next_seed_positions = []
        for seed_pos in seed_positions:
            # Sample probability that neighboring pixel will be visible
            prob_visible = np.random.rand(len(neighbors))
            for j, neighbor in enumerate(neighbors):
                if prob_visible[j] > threshold:
                    current_h, current_w = seed_pos
                    shift_h, shift_w = neighbor
                    # Ensure new height stays within image boundaries
                    new_h = max(min(current_h + shift_h, img_height - 1), 0)
                    # Ensure new width stays within image boundaries
                    new_w = max(min(current_w + shift_w, img_width - 1), 0)
                    # Update mask
                    mask[new_h, new_w] = 1
                    # Add new position to list of seeds
                    next_seed_positions.append((new_h, new_w))
        seed_positions = next_seed_positions

    return mask 

def random_blob_mask(image, min_val, max_val):
    '''
    Generates random blob masks
    
    Params:
    min_val, max_val: Lower and upper bound on number of iterations to be used for each blob. This will be sampled for each blob.
    '''
    im_shape = image.shape 

    mask = np.zeros(im_shape)
    num_iter = np.random.randint(min_val, max_val)
    threshold = 0.5+ np.random.random_sample()*0.25
    mask[:,:,0] = blob_mask(im_shape[:2], num_iter, threshold)

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask

def multi_blob_mask(im_shape, max_num_blobs, min_val, max_val, threshold):
    '''
    Returns a mask with one or more blobs

    Params:
    max_num_blobs: Maximum number of blobs. Number of blobs will be sampled between 1 and max_num_blobs.
    min_val, max_val: Lower and upper bound on number of iterations to be used for each blob. This will be sampled for each blob.
    threshold: Threshold used in each blob to either hide or make pixel visible. 
    '''
    mask = np.zeros(im_shape)

    # Sample number of blobs
    num_blobs = np.random.randint(1,max_num_blobs+1)
    # Generate the mask
    for _ in range(num_blobs):
        num_iter = np.random.randint(min_val, max_val)
        mask += blob_mask(im_shape, num_iter, threshold)

    mask[mask > 0] = 1
    return mask 

def random_multi_blob_mask(image, max_num_blobs, min_val, max_val):
    '''
    Generates random masks with one or more blobs

    Params:
    max_num_blobs: Maximum number of blobs. Number of blobs will be sampled between 1 and max_num_blobs.
    min_val, max_val: Lower and upper bound on number of iterations to be used for each blob. This will be sampled for each blob.
    '''
    im_shape = image.shape

    mask = np.zeros(im_shape)
    threshold =  0.5+ np.random.random_sample()*0.25
    mask[:,:,0] = multi_blob_mask(im_shape[:2], max_num_blobs, min_val, max_val, threshold)

    mask = mask.astype('uint8')
    cond_pixel_image = image*np.expand_dims(mask[:,:,0], axis = 2)
    mask = 255*mask

    return cond_pixel_image, mask 

def fixed_masking(fixed_mask):
    '''
    Returns a map function to return conditional pixels for all images from a fixed mask. 

    Params:
    fixed_mask: the mask (same shape as the images) to be used  
    '''
    def mask_map(image):
        '''
        Returns conditional pixels obtained from masking the images with a fixed mask and also appends the mask.
        Ex: If the input image has size [height,width,channels], the output will be of size [height,width,channels+1].
        The mask is appended as an extra color channel.

        Params:
        image: Image that has to be masked. Shape is [height, width, channels]

        '''
        mask = tf.constant(fixed_mask, dtype = tf.uint8)
        mask = tf.expand_dims(mask,axis = 2)
        cond_pixel_image = tf.multiply(image,mask)
        mask  = 255*mask 
        mask_image = tf.concat([cond_pixel_image,mask], axis = 2)

        return mask_image

    return mask_map

def random_masking(visible_prob):
    '''
    Returns a conditional pixel image from random masks and also appends the mask.
    Ex: If the input image has size [height,width,channels], the output will be of size [height,width,channels+1].
    The mask is appended as an extra color channel.

    Params:
    image: Image that has to be masked. Shape is [height, width, channels]

    '''
    def masking(image):
        im_shape = image.get_shape().as_list()

        mask = tf.random_uniform(im_shape[:-1], minval = 0, maxval = 1, dtype = tf.float32)
        mask = mask < visible_prob
        mask = tf.cast(mask, dtype = tf.uint8)
        mask = tf.expand_dims(mask,axis = 2)
        cond_pixel_image = tf.multiply(image,mask)
        mask  = 255*mask 
        mask_image = tf.concat([cond_pixel_image,mask], axis = 2)

        return mask_image

    return masking 

class DataSet(object):
    '''
    Generates a tf.data.Dataset object, where each element is (batch_image, batch_masked_image). 
    '''
    
    def __init__(self, images_list_path, num_epoch, batch_size, im_shape, num_channels, mask_type = 'bottom', mask_args = None, buffer_size = 200000, test_mode = False):
        '''
        Params:
        images_list_path: Path to the list of all image locations. 
        num_epoch: Number of epochs (int)
        batch_size: Batch size (int)
        im_shape: Input shape for the images. A list of 2 elements: [height, width]
        mask type: Specifying the type of mask to be used. 
        mask_args: Specifying details about the mask (Number of rows for bottom mask for example)
        test_mode: If true, dataset is not shuffled.  
        '''
        
        # Create a list of image paths
        input_file = open(images_list_path, 'r')
        self.record_list = []

        for line in input_file:
            line = line.strip()
            self.record_list.append(line)

        self.im_list = tf.constant(self.record_list)
        self.im_dataset = tf.data.Dataset.from_tensor_slices(self.im_list)
        self.im_dataset = self.im_dataset.map(im_preprocess(im_shape, num_channels))

        if mask_args is not None:
            mask_args = mask_args.split(',')
            mask_args = [float(x) for x in mask_args]

        if mask_type == 'bottom':
            # Fixed bottom mask with specified number of rows 
            if mask_args is not None:
                num_rows = int(mask_args[0])
            else:
                num_rows = 2
            mask = bottom_mask(im_shape, num_rows)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_bottom':
            # Random bottom mask 
            if mask_args is not None:
                min_val = int(mask_args[0])
                max_val = int(mask_args[1])
            else:
                min_val = 1
                max_val = 4
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_bottom_mask,[image, min_val, max_val], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'edge':
            # Edge mask
            if mask_args is not None:
                num_pixels = int(mask_args[0])
            else:
                num_pixels = 2
            mask = edge_mask(im_shape,num_pixels)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_edge':
            # Random edge mask
            if mask_args is not None:
                min_val = int(mask_args[0])
                max_val = int(mask_args[1])
            else:
                min_val = 1
                max_val = 4
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_edge_mask,[image, min_val, max_val], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'center':
            # Center mask
            if mask_args is not None:
                num_pixels = int(mask_args[0])
            else:
                num_pixels = 8
            mask = center_mask(im_shape,num_pixels)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_center':
            # Random center mask
            if mask_args is not None:
                min_val = int(mask_args[0])
                max_val = int(mask_args[1])
            else:
                min_val = 4
                max_val = 12
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_center_mask,[image, min_val, max_val], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'rectangle':
            # Rectangle mask (location random)
            if mask_args is not None:
                rect_height = int(mask_args[0])
                rect_width = int(mask_args[1])
            else:
                rect_height = 12
                rect_width = 8
            mask = rectangular_mask(im_shape,rect_height,rect_width)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_rectangle':
            # Random rectangle mask (size and location random)
            if mask_args is not None:
                max_height = int(mask_args[0])
                max_width = int(mask_args[1])
            else:
                max_height = 12
                max_width = 12
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_rectangular_mask,[image, max_height, max_width], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'blob':
            # Blob mask
            if mask_args is not None:
                num_iter = int(mask_args[0])
                threshold = mask_args[1]
            else:
                num_iter = 5
                threshold = 0.6
            mask = blob_mask(im_shape,num_iter,threshold)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_blob':
            # Random blob mask
            if mask_args is not None:
                min_val = int(mask_args[0])
                max_val = int(mask_args[1])
            else:
                min_val = 4
                max_val = 8
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_blob_mask,[image, min_val, max_val], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'multi_blob':
            # Multi blob mask
            if mask_args is not None:
                max_num_blobs = int(mask_args[0])
                min_val = int(mask_args[1])
                max_val = int(mask_args[2])
                threshold = mask_args[3]
            else:
                max_num_blobs = 3
                min_val = 4 
                max_val = 8 
                threshold = 0.6
            mask = multi_blob_mask(im_shape,max_num_blobs, min_val, max_val, threshold)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        elif mask_type == 'random_multi_blob':
            # Random multi blob mask
            if mask_args is not None:
                max_num_blobs = int(mask_args[0])
                min_val = int(mask_args[1])
                max_val = int(mask_args[2])
            else:
                max_num_blobs = 3
                min_val = 4
                max_val = 8
            self.masked_im_dataset = self.im_dataset.map(lambda image: [tf.py_func(random_multi_blob_mask,[image, max_num_blobs, min_val, max_val], (tf.uint8, tf.uint8))], num_parallel_calls = None)
            self.masked_im_dataset = self.masked_im_dataset.map(resize_func(im_shape,num_channels))

        elif mask_type == 'fixed_random':
            # Fixed random mask (low_val to high_val visible pixels)
            if mask_args is not None:
                low_val = int(mask_args[0])
                high_val = int(mask_args[1])
            else:
                low_val = 10
                high_val = 20
            num_visible = np.random.randint(low = low_val, high = high_val)
            print ("The number of visible pixels are {}".format(num_visible))
            mask = single_random_mask(im_shape, num_visible)
            self.masked_im_dataset = self.im_dataset.map(fixed_masking(mask), num_parallel_calls= None)

        else:
            # Random mask with a threshold 
            if mask_args is not None:
                visible_prob = mask_args[0]
            else:
                visible_prob = 0.15
            self.masked_im_dataset = self.im_dataset.map(random_masking(visible_prob), num_parallel_calls= None)

        self.dataset = tf.data.Dataset.zip((self.im_dataset,self.masked_im_dataset))

        if test_mode == False:
            self.dataset = self.dataset.shuffle(buffer_size = buffer_size)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat(num_epoch)















        
