from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from ops import *

class Net(object):
  def __init__(self, images, masked_images, num_colors, loss_func, scope):
    """
      images: [batch_size, height, width, in_channels] (float32)
      masked_images: [batch_size, height, width, in_channels+1] (float32)
      num_colors: Number of possible colors possible for each pixel (int32)
      loss func: Options are (i) weighted ,(ii) heuristic and (iii) vanilla_pcnn
    """
    with tf.variable_scope(scope) as scope:
      self.train = tf.placeholder(tf.bool)
      self.construct_net(images, masked_images, num_colors, loss_func)

  def prior_network(self, images, num_colors, num_gated=20):
    """
      images: [batch_size, height, width, in_channel]
      num_gated: Number of Gated PixelCNN convolutions 

    Returns:
      prior_logits: [batch_size, height, width, in_channel*num_colors]
    """
    batch_size, height, width, in_channel = images.get_shape().as_list()

    with tf.variable_scope('prior') as scope:
      # Special Gated Convolution
      inputs, state = special_gated_conv2d(images, 63, [7, 7], in_channel, scope="special_gated_conv")
      
      # Gated Convolutions
      for i in range(num_gated):
        inputs, state = gated_conv2d(inputs, state, [5, 5], in_channel, scope='gated' + str(i))

      if in_channel == 1:
        conv2 = conv2d(inputs, 1024, [1, 1], strides=[1, 1], scope="post_gated_conv")
        conv2 = tf.nn.relu(conv2)
        prior_logits = conv2d(conv2, in_channel * num_colors, [1, 1], strides=[1, 1], scope="logits_conv")

      if in_channel == 3:
        conv2 = conv2d_rgb(inputs, 1023, [1,1], strides = [1,1], scope = "post_gated_conv")
        conv2 = tf.nn.relu(conv2)
        prior_logits = conv2d_rgb(conv2, in_channel * num_colors, [1,1], strides=[1,1], scope="logits_conv")

      return prior_logits


  def conditioning_network(self, masked_images, num_colors, num_resnet=12):
    """
      masked_images: [batch_size, height, width, in_channels+1]
      num_resnet: Number of ResNet blocks

    Returns:
      conditioning_logits: [batch_size, height, width, in_channel*num_colors]
    """
    batch_size, height, width, in_channel = masked_images.get_shape().as_list()

    with tf.variable_scope('conditioning') as scope:
      inputs = masked_images
      inputs = conv2d(inputs, 64, [1, 1], strides=[1, 1], scope="conv_init")

      for i in range(num_resnet):
        inputs = resnet_block(inputs, 64, [3, 3], strides=[1, 1], scope='res' + str(i), train=self.train)
      
      conditioning_logits = conv2d(inputs, (in_channel - 1)*num_colors, [1, 1], strides=[1, 1], scope="logits_conv")

      return conditioning_logits

  def softmax_loss(self, logits, labels, num_colors):
    logits = tf.reshape(logits, [-1, num_colors])
    labels = tf.cast(labels, tf.int32)
    labels = tf.reshape(labels, [-1])
    return tf.losses.sparse_softmax_cross_entropy(
           labels, logits)

  def construct_net(self, images, masked_images, num_colors, loss_func = 'weighted'):
    """
    loss_func: Options are (i) weighted ,(ii) heuristic and (iii) vanilla_pcnn
    """

    #labels
    labels = tf.cast(images, tf.float32) * ((num_colors - 1.0)/255)
    
    #normalization images [-0.5, 0.5]
    images = tf.cast(images, tf.float32) / (255.0) - 0.5
    masked_images = tf.cast(masked_images, tf.float32) / (255.0) - 0.5
    self.prior_logits = self.prior_network(images, num_colors)
    self.conditioning_logits = self.conditioning_network(masked_images, num_colors)

    loss1 = self.softmax_loss(self.prior_logits + self.conditioning_logits, labels, num_colors)
    loss2 = self.softmax_loss(self.conditioning_logits, labels, num_colors)
    loss3 = self.softmax_loss(self.prior_logits, labels, num_colors)

    ##### Learn the weights to add the prior and conditional logits ######
    if loss_func == 'weighted':

      logits_shape = self.prior_logits.get_shape().as_list()
      W_add = tf.get_variable("W_add", shape = logits_shape[1:], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1))
      self.combined_logits = self.prior_logits + tf.multiply(self.conditioning_logits,W_add)
      self.loss = self.softmax_loss(self.combined_logits, labels)
      
      tf.summary.histogram("W_add",W_add)
      tf.summary.scalar('loss_prior', loss3)
      tf.summary.scalar('loss_conditional',loss2)
    

    #### Heuristic loss: loss = 2*conditional_loss + prior_loss
    if loss_func == 'heuristic':
      self.loss = loss3 + 2*loss2
      tf.summary.scalar('loss_prior', loss3)
      tf.summary.scalar('loss_conditional',loss2) 

    #### Vanilla RCNN loss
    if loss_func == 'vanilla_pcnn':
      self.loss = loss3

    tf.summary.scalar('loss', self.loss)
    

