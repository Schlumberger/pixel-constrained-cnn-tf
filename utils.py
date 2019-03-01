import numpy as np
import os
from skimage.io import imsave
from matplotlib.pyplot import get_cmap 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def save_samples(np_imgs, img_path):
  """
  Args:
    np_imgs: [N, H, W, C] float32
    img_path: str
  """
  np_imgs = np_imgs.astype(np.uint8)
  try:
    N, H, W, C = np_imgs.shape
  except ValueError:
    N, H, W = np_imgs.shape
    C = 1
    np_imgs = np.expand_dims(np_imgs,3)

  num = int(N ** (0.5))
  merge_img = np.zeros((num * H, num * W, C), dtype=np.uint8)
  for i in range(num):
    for j in range(num):
      merge_img[i*H:(i+1)*H, j*W:(j+1)*W, :] = np_imgs[i*num+j,:,:,:]

  if C == 1:
    merge_img = np.squeeze(merge_img,2)
    
  imsave(img_path, merge_img)

def save_test_samples(np_imgs, samples_dir = None, img_type = None, start_iter = None, epoch_no = None, record_list = None):
  """
  Args:
    np_imgs: [N, H, W, 3] float32
    samples_dir: str 
    img_type: str
    start_iter: int 
  """
  np_imgs = np_imgs.astype(np.uint8)
  #print (np_imgs.shape)
  N, H, W, C = np_imgs.shape

  if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

  for i in range(N):
    step = start_iter + i
    name = record_list[step]
    start_idx = name.rindex('/')
    end_idx = name.rindex('.')
    name = name[start_idx+1:end_idx]

    if img_type == 'generated':
      img_path = samples_dir + '/' + name + '_' + img_type + '_' + str(epoch_no) + '.jpg'
      img = np_imgs[i,:,:,:]
    elif img_type == 'masked':
      img_path = samples_dir + '/' + name + '_' + img_type + '.jpg'
      img = np_imgs[i,:,:,:-1]
      mask = np_imgs[i,:,:,-1]
      mask = np.stack((np.squeeze(mask),)*(C-1), axis = -1)
      img[mask == 0] = 100 
    else:
      img_path = samples_dir + '/' + name + '_' + img_type + '.jpg'
      img = np_imgs[i,:,:,:]

    if np.shape(img)[2] == 1:
      img = np.squeeze(img,2)
    imsave(img_path, img)

def save_logits_image(logits_imgs, mask_imgs, cmap = 'jet', row_num = None, samples_dir = None, start_iter = None, epoch_no = None, record_list = None):
  """
  Args:
    logits_imgs: [N,h,w,c*1] float32
    mask_imgs: [N,h,w,c+1]
  """ 
  N,H,W,C = logits_imgs.shape
  convert_to_cmap = get_cmap(cmap)

  if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

  for i in range(N):
    step = start_iter + i
    name = record_list[step]
    start_idx = name.rindex('/')
    end_idx = name.rindex('.')
    name = name[start_idx+1:end_idx]

    img_path = samples_dir + '/' + name + '_' + 'probs' + '_' + str(epoch_no) + '_' + str(row_num) + '.jpg'
    rgb_logits_im = convert_to_cmap(logits_imgs[i,:,:,0])
    img = np.delete(rgb_logits_im,3,2)

    masked_im = np.squeeze(mask_imgs[i,:,:,:-1])/255.0
    mask = mask_imgs[i,:,:,-1]
    img[:,:,0][mask > 0] = masked_im[mask > 0]
    img[:,:,1][mask > 0] = masked_im[mask > 0]
    img[:,:,2][mask > 0] = masked_im[mask > 0]

    imsave(img_path, img)

  

def logits_2_pixel_value(logits, mu=1.1, sample_opt = 'weight_avg'):
  """
  Args:
    logits: [n, 256] float32
    mu    : float32
  Returns:
    pixels: [n] float32
  """
  n,num_colors = np.shape(logits)
  rebalance_logits = logits * mu
  probs = softmax(rebalance_logits)
  pixel_dict = np.arange(0, num_colors, dtype=np.float32)

  if sample_opt == 'weight_avg':
    pixels = np.sum(probs * pixel_dict, axis=1)
  if sample_opt == 'prob_sample':
    pixels = np.ones((n,), dtype = np.int)
    for i in range(n):
      pixels[i] = np.random.choice(pixel_dict, p = probs[i,:])

  pixel_probs = probs[np.arange(0,n),np.floor(pixels).astype(int)]
  pixels = pixels * (255/(num_colors - 1.0))

  return np.floor(pixels), pixel_probs


def log_likelihood_plot(samples,log_likelihoods, ncols, cmap = 'jet', samples_dir = None, record_list = None):
  """
  Sorts samples by their log likelihoods and creates an image representing the log likelihood of each sample 
  as a boxwith color and size proportional to the log likelihood. 

  Args:
    samples: [num_elements, num_samples, height, width, num_channels] float32
    log_likelihoods: [num_elements, num_samples]
    ncols: Number of columns in the final image
  """

  num_elements = samples.shape[0]
  num_samples = samples.shape[1]
  height = samples.shape[2]
  width = samples.shape[3]
  num_channels = samples.shape[4]

  if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

  nrows = num_samples // ncols 
  assert num_samples == nrows*ncols, "Number of samples not equal to nrows*ncols"

  for i in range(num_elements):
    elm_samples = samples[i]
    elm_likelihoods = log_likelihoods[i,:]
    # Sorted by negative log likelihood
    sorted_indices = np.argsort(-elm_likelihoods)
    sorted_likelihoods = elm_likelihoods[sorted_indices]
    sorted_samples = elm_samples[sorted_indices]
    # Normalize log likelihoods to be in 0-1 range
    min_ll, max_ll = sorted_likelihoods[-1], sorted_likelihoods[0]
    normalized_likelihoods = (sorted_likelihoods - min_ll)/(max_ll - min_ll)

    # For each sample, draw an image with a box proportional in size and color to the log-likelihood value
    ll_images = np.ones((num_samples,height,width,3))
    # Specify box sizes 
    lower_width = width//2 - width//5
    upper_width = width//2 + width//5 
    max_box_height = height
    min_box_height = 1
    # Generate colors for the boxes
    convert_to_cmap = get_cmap(cmap)
    # Remove alpha channel from colormap
    colors = convert_to_cmap(normalized_likelihoods)[:,:-1]

    # Fill out images with boxes
    for sample_no in range(num_samples):
      norm_ll = normalized_likelihoods[sample_no]
      box_height = int(min_box_height + (max_box_height - min_box_height)*norm_ll)
      box_color = colors[sample_no]
      for j in range(3):
        ll_images[sample_no,height- box_height:height, lower_width:upper_width, j] = box_color[j]

    # Save images and boxes in decreasing order of log_likelihood 
    img = np.zeros((2*num_samples,height,width,3))
    if num_channels == 1:
      img[0::2] = np.squeeze(np.stack((sorted_samples,)*3, axis=3))
    else:
      img[0::2] = sorted_samples
    img[1::2] = ll_images*255 
    img = np.reshape(img,(num_samples,2*height,width,3))

    final_img = (img.reshape(nrows,ncols,2*height,width,3)
                 .swapaxes(1,2)
                 .reshape(2*height*nrows,width*ncols,3)).astype('uint8')

    name = record_list[i]
    start_idx = name.rindex('/')
    end_idx = name.rindex('.')
    name = name[start_idx+1:end_idx]
    final_img_path = samples_dir + '/' + name + '_' + 'generated_likelihood.jpg'

    imsave(final_img_path, final_img)
