from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *
from data import *
from net import *
from utils import *
import os
import time

flags = tf.app.flags
conf = flags.FLAGS

class Solver(object):
  def __init__(self):
    # Directory params
    self.train_dir = conf.train_dir
    self.samples_dir = conf.samples_dir
    self.sample_opt = conf.sample_opt
    self.imgs_list_path = conf.imgs_list_path

    if not os.path.exists(self.train_dir):
      os.makedirs(self.train_dir)
    if not os.path.exists(self.samples_dir):
      os.makedirs(self.samples_dir)    

    # Training params
    self.num_epoch = conf.num_epoch
    self.batch_size = conf.batch_size
    self.learning_rate = conf.learning_rate
    self.decay_step = conf.decay_step
    self.decay_rate = conf.decay_rate
    self.staircase = conf.staircase
    
    # Dataset params
    self.im_shape = list(map(int, conf.im_shape.split(",")))
    self.mask_type = conf.mask_type
    self.mask_args = conf.mask_args
    self.num_channels = conf.num_channels
    self.num_colors = conf.num_colors
    self.buffer_size = conf.buffer_size

    # Other params
    self.loss_func = conf.loss_func
    self.num_checkpoints = conf.num_checkpoints
    self.sample_every = conf.sample_every
    self.checkpoint_every = conf.checkpoint_every
    self.mu = conf.sample_temp

    # Optimizer params
    self.device_id = conf.device_id
    if conf.use_gpu:
      device_str = '/gpu:' +  str(self.device_id)
    else:
      device_str = '/cpu:0'

    with tf.device(device_str):
      #dataset
      self.data = DataSet(self.imgs_list_path, self.num_epoch, self.batch_size, self.im_shape, self.num_channels, self.mask_type, self.mask_args, self.buffer_size, test_mode = False)

      self.handle = tf.placeholder(tf.string, shape=[])
      iterator = tf.data.Iterator.from_string_handle(self.handle, self.data.dataset.output_types, self.data.dataset.output_shapes)
      self.images, self.masked_images = iterator.get_next()
      self.train_iterator = self.data.dataset.make_one_shot_iterator()

      #network
      self.net = Net(self.images, self.masked_images, self.num_colors, self.loss_func, 'pattern_model')

      #optimizer
      self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
      learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_step, self.decay_rate, staircase=self.staircase)
      #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9, epsilon=1e-8)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      self.train_op = optimizer.minimize(self.net.loss, global_step=self.global_step)

  def train(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep = self.num_checkpoints)

    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

    train_handle = sess.run(self.train_iterator.string_handle())
    
    iters = 0
    while True:
      try:
        iters += 1

        if iters % self.sample_every == 0 or iters % 10 == 0:
          
          # Run training steps 
          t1 = time.time()
          _, loss, np_images, np_masked_images = sess.run([self.train_op, self.net.loss, self.images, self.masked_images], feed_dict={self.handle: train_handle, self.net.train: True})
          t2 = time.time()
          print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))

          # Sample from training data
          if iters % self.sample_every == 0:
            print ("Sampling")
            self.sample(sess, np_images, np_masked_images, mu=self.mu, step=iters)

          # Write summary
          if iters % 10 == 0:
            summary_str = sess.run(summary_op, feed_dict={self.net.train: True, self.images: np_images, self.masked_images: np_masked_images})
            summary_writer.add_summary(summary_str, iters)
        
        else:
          t1 = time.time()
          _, loss = sess.run([self.train_op, self.net.loss], feed_dict={self.handle: train_handle, self.net.train: True})
          t2 = time.time()
          print('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' % ((iters, loss, self.batch_size/(t2-t1), (t2-t1))))
        
        # Save checkpoints
        if iters % self.checkpoint_every == 0:
          checkpoint_path = os.path.join(self.train_dir, 'model')
          saver.save(sess, checkpoint_path, global_step=iters)

      except tf.errors.OutOfRangeError:
        checkpoint_path = os.path.join(self.train_dir, 'model')
        saver.save(sess, checkpoint_path)
        print('Done training -- epoch limit reached')
        break
     
    
  def sample(self, sess, np_images, np_masked_images, mu=1.1, step=None):

    # Logits
    cond_logits = self.net.conditioning_logits
    prior_logits = self.net.prior_logits

    #Images
    batch_size, height, width, num_channels = np_images.shape 
    gen_images = np.zeros((np_images.shape), dtype=np.float32)
    
    if self.loss_func == 'heuristic':
      np_cond_logits = sess.run(cond_logits, feed_dict={self.masked_images:np_masked_images, self.net.train:False})
    
    if self.mask_type in ['fixed_random','bottom','edge','center','rectangle','blob']:

      mask = np_masked_images[0,:,:,-1]

      for i in range(height):
        for j in range(width):

          if mask[i,j] > 0: 
            gen_images[:, i, j, :] = np_masked_images[:,i,j,:-1]

          else:
            for c in range(num_channels):
              if self.loss_func == 'heuristic':
                np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
                new_pixel, _ = logits_2_pixel_value(2*np_cond_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors] + np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:, i, j, c] = new_pixel
              
              if self.loss_func == 'weighted':
                np_comb_logits = sess.run(self.net.combined_logits, feed_dict={self.images: gen_images, self.masked_images:np_masked_images, self.net.train: False})
                new_pixel, _ = logits_2_pixel_value(np_comb_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:,i,j,c] = new_pixel

              if self.loss_func == 'vanilla_pcnn':
                np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
                new_pixel, _ = logits_2_pixel_value(np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:, i, j, c] = new_pixel

    else:

      for i in range(height):
        for j in range(width):

          mask_false = np_masked_images[:,i,j,-1] > 0 

          for c in range(num_channels):
            if self.loss_func == 'heuristic':
              np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
              new_pixel, _ = logits_2_pixel_value(2*np_cond_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors] + np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
              gen_images[:, i, j, c] = new_pixel

            if self.loss_func == 'weighted':
              np_comb_logits = sess.run(self.net.combined_logits, feed_dict={self.images: gen_images, self.masked_images:np_masked_images, self.net.train: False})
              new_pixel, _ = logits_2_pixel_value(np_comb_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
              gen_images[:,i,j,c] = new_pixel

            gen_images[:,i,j,c][mask_false] = np_masked_images[:,i,j,c][mask_false]


   
    # Save masked, original and generated images 
    save_samples(np_masked_images[:,:,:,-1], self.samples_dir + '/masked' + '_' + str(step) + '.jpg')
    save_samples(np_images, self.samples_dir + '/original'  + '_' + str(step) + '.jpg')
    save_samples(gen_images, self.samples_dir + '/generated' + '_' + str(step) + '.jpg')


################################################################ Test Sample Mode ###########################################################################


class test_sample(object):
  '''
  Load model, sample a image from a masked_image and store generated images. 
  '''
  def __init__(self):
    # Directory params
    self.samples_dir = conf.samples_dir
    self.model_name = conf.model_name 
    self.sample_opt = conf.sample_opt
    self.imgs_list_path = conf.imgs_list_path

    if not os.path.exists(self.samples_dir):
      os.makedirs(self.samples_dir)    

    # Dataset params
    self.num_epoch = conf.num_samples
    self.batch_size = conf.batch_size
    self.im_shape = list(map(int, conf.im_shape.split(",")))
    self.mask_type = conf.mask_type
    self.mask_args = conf.mask_args
    self.num_channels = conf.num_channels
    self.num_colors = conf.num_colors

    # Output params
    '''
    gen_type options : (i) sample, (ii) uncertainty and (iii) logits
    ''' 
    self.gen_type = conf.gen_type

    # Other params
    self.loss_func = conf.loss_func
    self.mu = conf.sample_temp

    # Optimizer params
    self.device_id = conf.device_id
    if conf.use_gpu:
      device_str = '/gpu:' +  str(self.device_id)
    else:
      device_str = '/cpu:0'

    with tf.device(device_str):
      #dataset
      self.data = DataSet(self.imgs_list_path, self.num_epoch, self.batch_size, self.im_shape, self.num_channels, self.mask_type, self.mask_args, test_mode = True)

      self.handle = tf.placeholder(tf.string, shape=[])
      iterator = tf.data.Iterator.from_string_handle(self.handle, self.data.dataset.output_types, self.data.dataset.output_shapes)
      self.images, self.masked_images = iterator.get_next()
      self.test_iterator = self.data.dataset.make_one_shot_iterator()

      #network 
      self.net = Net(self.images, self.masked_images, self.num_colors, self.loss_func, 'pattern_model') 

  def test(self):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, self.model_name)
    test_handle = sess.run(self.test_iterator.string_handle())
    print ("Generating images from masked_images in test folder using model: {}" .format(self.model_name))

    num_elements = len(self.data.record_list)
    if self.gen_type == 'logits':
      if self.num_colors != 2:
        raise(RuntimeError("Logits generation only works for models with 2 colors. Current model has {} colors.".format(self.num_colors)))

    if self.gen_type == 'uncertainty':
      samples = np.zeros((num_elements, self.num_epoch, self.im_shape[0], self.im_shape[1],self.num_channels), dtype = np.float32)
      log_likehoods = np.zeros((num_elements, self.num_epoch))

    epoch_masked_images = np.zeros((num_elements, self.im_shape[0], self.im_shape[1], self.num_channels+1))

    i = 0
    j = 0
    while True:
      try:
        if j == 0:
          np_images, np_masked_images = sess.run([self.images, self.masked_images], feed_dict={self.handle: test_handle, self.net.train: False})
          gen_images, log_probs = self.sample(sess, np_images, np_masked_images, mu=self.mu, it = i, ep = j, gen_type = self.gen_type)
          epoch_masked_images[i:i+np_images.shape[0],:,:,:] = np_masked_images
        else:
          np_images = sess.run(self.images, feed_dict={self.handle: test_handle, self.net.train: False})
          np_masked_images = epoch_masked_images[i:i+np_images.shape[0],:,:,:]
          gen_images, log_probs = self.sample(sess, np_images, np_masked_images, mu=self.mu, it = i, ep = j, gen_type = self.gen_type)

        print ("Sampling batch no: {}, Sampling epoch no: {}".format((1 + (i/self.batch_size)),j))

        if self.gen_type == 'uncertainty':
          samples[i:i+np_images.shape[0],j,:,:,:] = gen_images
          log_likehoods[i:i+np_images.shape[0],j] = log_probs
        
        i = i + np_images.shape[0]
        if i == num_elements:
          i = 0
          j = j + 1

      except tf.errors.OutOfRangeError:
        break

    if self.gen_type == 'uncertainty':
      ncols = 8
      log_likelihood_plot(samples,log_likehoods,ncols,samples_dir = self.samples_dir + '/generated', record_list = self.data.record_list)


  def sample(self, sess, np_images, np_masked_images, mu=1.1, it=None, ep=None, gen_type = 'sample'):

    # Logits
    cond_logits = self.net.conditioning_logits
    prior_logits = self.net.prior_logits

    # Images
    batch_size, height, width, num_channels = np_images.shape 
    gen_images = np.zeros((np_images.shape), dtype=np.float32)
 
    # log likelihood
    log_probs = np.zeros(batch_size) 

    if self.loss_func == 'heuristic':
      np_cond_logits = sess.run(cond_logits, feed_dict={self.masked_images:np_masked_images, self.net.train:False})
    
    if self.mask_type in ['fixed_random','bottom','edge','center','rectangle','blob']:

      mask = np_masked_images[0,:,:,-1]

      for i in range(height):
        for j in range(width):

          if mask[i,j] > 0: 
            gen_images[:, i, j, :] = np_masked_images[:,i,j,:-1]

          else:
            for c in range(num_channels):
              if self.loss_func == 'heuristic':
                np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
                new_pixel, pixel_probs = logits_2_pixel_value(2*np_cond_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors] + np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:, i, j, c] = new_pixel
                log_probs += np.log(pixel_probs + 1e-9)

                if gen_type == 'logits':
                  total_logits = 2*np_cond_logits + np_prior_logits
                  total_probs = softmax(total_logits)
              
              if self.loss_func == 'weighted':
                np_comb_logits = sess.run(self.net.combined_logits, feed_dict={self.images: gen_images, self.masked_images:np_masked_images, self.net.train: False})
                new_pixel, pixel_probs = logits_2_pixel_value(np_comb_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:,i,j,c] = new_pixel
                log_probs += np.log(pixel_probs + 1e-9)

                if gen_type == 'logits': 
                  total_logits = np_comb_logits
                  total_probs = softmax(total_logits)
               
              if self.loss_func == 'vanilla_pcnn':
                np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
                new_pixel, pixel_probs = logits_2_pixel_value(np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
                gen_images[:, i, j, c] = new_pixel
                log_probs += np.log(pixel_probs + 1e-9)

                if gen_type == 'logits':
                  total_logits = np_prior_logits
                  total_probs = softmax(total_logits)

        # Generating logits
        if gen_type == 'logits':
          new_logits_im = np.zeros((np_images.shape), dtype = np.float32)
          new_mask = np.zeros((np_masked_images.shape), dtype = np.float32)
          # Rows from 0 to i
          new_logits_im[:,0:i+1,:,:] = gen_images[:,0:i+1,:,:]/255.
          new_mask[:,0:i+1,:,:-1] = gen_images[:,0:i+1,:,:]
          new_mask[:,0:i+1,:,-1] = 1.0
          # Rows from i to height
          new_logits_im[:,i:height,:,:] = total_probs[:,i:height,:,1::2]
          new_mask[:,i:height,:,:] = np_masked_images[:,i:height,:,:]

          # print the logits image
          save_logits_image(new_logits_im, new_mask, cmap = 'jet', row_num = i+1, samples_dir = self.samples_dir + 'probs_images', start_iter = it, 
                            epoch_no = ep, record_list = self.data.record_list)

    else:

      for i in range(height):
        for j in range(width):

          mask_false = np_masked_images[:,i,j,-1] > 0 

          for c in range(num_channels):
            if self.loss_func == 'heuristic':
              np_prior_logits = sess.run(prior_logits, feed_dict={self.images: gen_images})
              new_pixel, pixel_probs = logits_2_pixel_value(2*np_cond_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors] + np_prior_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
              gen_images[:, i, j, c] = new_pixel

              if gen_type == 'logits':
                total_logits = 2*np_cond_logits + np_prior_logits
                total_probs = softmax(total_logits)
            
            if self.loss_func == 'weighted':
              np_comb_logits = sess.run(self.net.combined_logits, feed_dict={self.images: gen_images, self.masked_images:np_masked_images, self.net.train: False})
              new_pixel, pixel_probs = logits_2_pixel_value(np_comb_logits[:, i, j, c*self.num_colors:(c+1)*self.num_colors], mu=mu, sample_opt = self.sample_opt)
              gen_images[:,i,j,c] = new_pixel

              if gen_type == 'logits':
                total_logits = np_comb_logits
                total_probs = softmax(total_logits)

            gen_images[:,i,j,c][mask_false] = np_masked_images[:,i,j,c][mask_false]
            pixel_probs[mask_false] = 1
            log_probs += np.log(pixel_probs + 1e-9)

        # Generating logits
        if gen_type == 'logits':
          new_logits_im = np.zeros((np_images.shape), dtype = np.float32)
          new_mask = np.zeros((np_masked_images.shape), dtype = np.float32)
          # Rows from 0 to i
          new_logits_im[:,0:i+1,:,:] = gen_images[:,0:i+1,:,:]/255.0
          new_mask[:,0:i+1,:,:-1] = gen_images[:,0:i+1,:,:]
          new_mask[:,0:i+1,:,-1] = 1.0
          # Rows from i to height
          new_logits_im[:,i:height,:,:] = total_probs[:,i:height,:,1::2]
          new_mask[:,i:height,:,:] = np_masked_images[:,i:height,:,:]
          
          # print the logits image
          save_logits_image(new_logits_im, new_mask, cmap = 'jet', row_num = i+1, samples_dir = self.samples_dir + '/probs_images', start_iter = it, 
                            epoch_no = ep, record_list = self.data.record_list)
    
    # Save masked, original and generated images 
    if ep == 0:
      save_test_samples(np_masked_images[:,:,:,:], samples_dir = self.samples_dir + '/masked', img_type = 'masked', start_iter = it, epoch_no = ep, record_list = self.data.record_list)
      save_test_samples(np_images, samples_dir = self.samples_dir + '/original' , img_type = 'original', start_iter = it, epoch_no = ep, record_list = self.data.record_list)

    if gen_type == 'sample': 
      save_test_samples(gen_images, samples_dir = self.samples_dir + '/generated', img_type = 'generated', start_iter = it, epoch_no = ep, record_list = self.data.record_list)

    return gen_images, log_probs



































