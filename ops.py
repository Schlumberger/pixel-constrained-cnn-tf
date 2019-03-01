from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2d"):
  """
  Args:
    inputs: nhwc
    kernel_shape: [height, width]
    mask_type: 
      (i) None
      (ii) sp_horizontal
      (iii) sp_vertical
      (iv) horizontal
      (v) vertical

  Returns:
    outputs: nhwc
  """
  with tf.variable_scope(scope) as scope:
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    batch_size, height, width, in_channel = inputs.get_shape().as_list()

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width must be odd number"
    mask = np.zeros((kernel_h, kernel_w, in_channel, num_outputs), dtype=np.float32)
    
    if mask_type == 'sp_horizontal':
      mask[center_h,:center_w,:,:] = 1

    elif mask_type == 'sp_vertical':
      mask[:center_h,:,:,:] = 1

    elif mask_type == 'horizontal':
      mask[center_h,:center_w+1,:,:] = 1

    elif mask_type == 'vertical':
      mask[:center_h+1,:,:,:] = 1

    else:
      mask[:,:,:,:] = 1 

    weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
    weights = tf.get_variable("weights", weights_shape,
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))
    weights = weights * mask
    biases = tf.get_variable("biases", [num_outputs],
          tf.float32, tf.constant_initializer(0.0))

    outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding="SAME")
    outputs = tf.nn.bias_add(outputs, biases)

    return outputs

def conv2d_rgb(inputs, num_outputs, kernel_shape, strides = [1,1], mask_type = None, scope = 'conv2d_rgb'):
  """
   Args:
    inputs: nhwc
    kernel_shape: [height, width]
    mask_type: 
      (i) None
      (ii) sp_horizontal
      (iii) sp_vertical
      (iv) horizontal
      (v) vertical

  Returns:
    outputs: nhwc
  """
  with tf.variable_scope(scope) as scope:
    kernel_h, kernel_w = kernel_shape
    stride_h, stride_w = strides
    batch_size, height, width, in_channel = inputs.get_shape().as_list()

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    assert kernel_h % 2 == 1 and kernel_w % 2 == 1, "kernel height and width must be odd number"

    r_mask = np.zeros((kernel_h, kernel_w, in_channel, int(num_outputs/3)), dtype=np.float32)
    g_mask = np.zeros((kernel_h, kernel_w, in_channel, int(num_outputs/3)), dtype=np.float32)
    b_mask = np.zeros((kernel_h, kernel_w, in_channel, int(num_outputs/3)), dtype=np.float32)

    if mask_type == 'sp_horizontal':
      # Red mask
      r_mask[center_h,:center_w,:,:] = 1

      # Green mask
      g_mask[center_h,:center_w,:,:] = 1
      g_mask[center_h,center_w,:1,:] = 1

      # Blue mask
      b_mask[center_h,:center_w,:,:] = 1
      b_mask[center_h,center_w,:2,:] = 1

    elif mask_type == 'horizontal':
      # Red mask
      r_mask[center_h,:center_w,:,:] = 1
      r_mask[center_h,center_w,:int(in_channel/3),:] = 1

      # Green mask
      g_mask[center_h,:center_w,:,:] = 1
      g_mask[center_h,center_w,:2*int(in_channel/3),:] = 1

      # Blue mask
      b_mask[center_h,:center_w+1,:,:] = 1

    else:
      r_mask[:,:,:int(in_channel/3),:] = 1 
      g_mask[:,:,:2*int(in_channel/3),:] = 1
      b_mask[:,:,:,:] = 1

    # Red convolutions
    r_weights_shape = [kernel_h,kernel_w,in_channel,int(num_outputs/3)]
    r_weights = tf.get_variable("r_weights", r_weights_shape, 
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))

    r_weights = r_weights * r_mask
    r_biases = tf.get_variable("r_biases", [int(num_outputs/3)],
          tf.float32, tf.constant_initializer(0.0))

    r_outputs = tf.nn.conv2d(inputs,r_weights, [1,stride_h,stride_w,1], padding = "SAME")
    r_outputs = tf.nn.bias_add(r_outputs,r_biases)

    # Green convolutions
    g_weights_shape = [kernel_h,kernel_w,in_channel,int(num_outputs/3)]
    g_weights = tf.get_variable("g_weights", g_weights_shape, 
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))

    g_weights = g_weights * g_mask
    g_biases = tf.get_variable("g_biases", [int(num_outputs/3)],
          tf.float32, tf.constant_initializer(0.0))

    g_outputs = tf.nn.conv2d(inputs,g_weights, [1,stride_h,stride_w,1], padding = "SAME")
    g_outputs = tf.nn.bias_add(g_outputs,g_biases)

    # Blue convolutions
    b_weights_shape = [kernel_h,kernel_w,in_channel,int(num_outputs/3)]
    b_weights = tf.get_variable("b_weights", b_weights_shape, 
      tf.float32, tf.truncated_normal_initializer(stddev=0.1))

    b_weights = b_weights * b_mask
    b_biases = tf.get_variable("b_biases", [int(num_outputs/3)],
          tf.float32, tf.constant_initializer(0.0))

    b_outputs = tf.nn.conv2d(inputs,b_weights, [1,stride_h,stride_w,1], padding = "SAME")
    b_outputs = tf.nn.bias_add(b_outputs,b_biases)

    # Concatenate outputs
    outputs = tf.concat([r_outputs,g_outputs,b_outputs],axis = 3, name = 'concat_outputs')

    return outputs


def gated_conv2d(inputs, state, kernel_shape, num_channels, scope):
  """
  Args:
    inputs: nhwc
    state:  nhwc
    kernel_shape: [height, width]
  Returns:
    outputs: nhwc
    new_state: nhwc
  """
  with tf.variable_scope(scope) as scope:
    batch_size, height, width, in_channel = inputs.get_shape().as_list()
    kernel_h, kernel_w = kernel_shape

    #state route (vertical mask)
    left = conv2d(state, 2 * in_channel, kernel_shape, strides=[1, 1], mask_type='vertical', scope="conv_s1")
    left1 = left[:, :, :, 0:in_channel]
    left2 = left[:, :, :, in_channel:]
    left1 = tf.nn.tanh(left1)
    left2 = tf.nn.sigmoid(left2)
    new_state = left1 * left2

    # Vertical ----> Horizontal
    left2right = conv2d(left, 2 * in_channel, [1, 1], strides=[1, 1], scope="conv_s2")

    # For num_channels == 1
    if num_channels == 1:
      #input route (horizontal mask)
      right = conv2d(inputs, 2 * in_channel, [1, kernel_w], strides=[1, 1], mask_type='horizontal', scope="conv_r1")
      right = right + left2right
      right1 = right[:, :, :, 0:in_channel]
      right2 = right[:, :, :, in_channel:]
      right1 = tf.nn.tanh(right1)
      right2 = tf.nn.sigmoid(right2)
      up_right = right1 * right2
      up_right = conv2d(up_right, in_channel, [1, 1], strides=[1, 1], scope="conv_r2")
      outputs = inputs + up_right

    # For num_channels == 3(rgb)
    if num_channels == 3:
      #input route (horizontal mask)
      right1 = conv2d_rgb(inputs, in_channel, [1,kernel_w], strides = [1,1], mask_type = 'horizontal', scope = "conv_r11")
      right2 = conv2d_rgb(inputs, in_channel, [1,kernel_w], strides = [1,1], mask_type = 'horizontal', scope = "conv_r12")
      right1 = tf.nn.tanh(right1 + left2right[:,:,:,0:in_channel])
      right2 = tf.nn.sigmoid(right2 + left2right[:,:,:,in_channel:])
      up_right = right1 * right2
      up_right = conv2d_rgb(up_right, in_channel, [1, 1], strides=[1, 1], scope="conv_r2")
      outputs = inputs + up_right

    return outputs, new_state


def special_gated_conv2d(inputs, num_outputs, kernel_shape, num_channels, scope):
  """
  Args:
    inputs: nhwc
    kernel_shape: [height, width]

  Returns:
    outputs: nhwc
    new_state: nhwc
  """

  with tf.variable_scope(scope) as scope:
    batch_size, height, width, in_channel = inputs.get_shape().as_list()
    in_channel = num_outputs
    kernel_h, kernel_w = kernel_shape

    # State route (special vertical mask)
    left = conv2d(inputs, 2*in_channel, kernel_shape, strides=[1,1], mask_type = 'sp_vertical', scope = "conv_s1")
    left1 = left[:, :, :, 0:in_channel]
    left2 = left[:, :, :, in_channel:]
    left1 = tf.nn.tanh(left1)
    left2 = tf.nn.sigmoid(left2)
    new_state = left1 * left2

    # Vertical ----> Horizontal
    left2right = conv2d(left, 2 * in_channel, [1, 1], strides=[1, 1], scope="conv_s2")

    # For num_channels == 1
    if num_channels == 1: 
      # Input route (special horizontal mask)
      right = conv2d(inputs, 2*in_channel, [1,kernel_w], strides=[1,1], mask_type = 'sp_horizontal', scope = "conv_r1")
      right = right + left2right
      right1 = right[:, :, :, 0:in_channel]
      right2 = right[:, :, :, in_channel:]
      right1 = tf.nn.tanh(right1)
      right2 = tf.nn.sigmoid(right2)
      up_right = right1 * right2
      up_right = conv2d(up_right, in_channel, [1, 1], strides=[1, 1], scope="conv_r2")
      outputs = up_right 

    # For num_channels == 3
    if num_channels == 3:
      # Input route (special horizontal mask)
      right1 = conv2d_rgb(inputs, in_channel, [1,kernel_w], strides = [1,1], mask_type = 'sp_horizontal', scope = "conv_r11")
      right2 = conv2d_rgb(inputs, in_channel, [1,kernel_w], strides = [1,1], mask_type = 'sp_horizontal', scope = "conv_r12")
      right1 = tf.nn.tanh(right1 + left2right[:,:,:,0:in_channel])
      right2 = tf.nn.sigmoid(right2 + left2right[:,:,:,in_channel:])
      up_right = right1 * right2
      up_right = conv2d_rgb(up_right, in_channel, [1, 1], strides=[1, 1], scope="conv_r2")
      outputs = up_right

    return outputs, new_state

def batch_norm(x, train=True, scope=None):
  return tf.contrib.layers.batch_norm(x, center=True, scale=True, updates_collections=None, is_training=train, trainable=True, scope=scope)

def resnet_block(inputs, num_outputs, kernel_shape, strides=[1, 1], scope=None, train=True):
  """
  Args:
    inputs: nhwc
    num_outputs: int
    kernel_shape: [kernel_h, kernel_w]
  Returns:
    outputs: nhw(num_outputs)
  """
  with tf.variable_scope(scope) as scope:
    conv1 = conv2d(inputs, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv1")
    bn1 = batch_norm(conv1, train=train, scope='bn1')
    relu1 = tf.nn.relu(bn1)
    conv2 = conv2d(relu1, num_outputs, kernel_shape, strides=[1, 1], mask_type=None, scope="conv2")
    bn2 = batch_norm(conv2, train=train, scope='bn2')
    output = inputs + bn2

    return output 

