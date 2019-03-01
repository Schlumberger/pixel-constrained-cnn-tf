import tensorflow as tf
import sys
sys.path.insert(0, './')
from solver import *

flags = tf.app.flags

# Directory params
flags.DEFINE_string("train_dir", "models", "trained model save path")
flags.DEFINE_string("samples_dir", "samples", "sampled images save path")
flags.DEFINE_string("imgs_list_path", "data/train.txt", "images list file path")

# Training params
flags.DEFINE_integer("num_epoch", 30, "train epoch num")
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_float("learning_rate", 4e-4, "learning rate")
flags.DEFINE_boolean("staircase", False, "If True, decay learn_rt at discrete intervals (default: False)")
flags.DEFINE_integer("decay_step", 500000, "Step after learn_rt is decayed (default: 500000)")
flags.DEFINE_float("decay_rate", 0.9, "Decay rate (default: 0.9)")
flags.DEFINE_string("sample_opt", "prob_sample", "method to sample new pixels, options are (i)prob_sample and (ii)weight_avg")


# Dataset params
flags.DEFINE_string("mask_type", 'blob', "Mask type to be used (default: blob)")
flags.DEFINE_string("mask_args", None, "Details about the mask (For ex. number of rows for bottom mask)")
'''
Mask types and corresponding arguments (separated by a comma if more than 1 arg):
'bottom': 1 arg specifying number of bottom rows (default is '2')
'random_bottom': 2 args specifying minimum and maximum number of bottom rows (default is '1,4')
'edge': 1 arg specifying number of edge pixels (default is '2')
'random_edge': 2 args specifying minimum and maximum number of edge pixels (default is '1,4')
'center': 1 arg specifying the size of center square (default is '8')
'random_center': 2 args specifying the minimum and maximum size of center square (default is '4,12')
'rectangle': 2 args specifying the height and width of the rectangle (default is '12,8')
'random_rectangle': 2 args specifying the max_height and max_width of the rectangle (default is '12,12')
'blob': 2 args specifying number of interations and threshold to generate a blob (default is '5,0.6')
'random_blob': 2 args specifying the minimum and maximum number of iterations to generate a blob (default is '4,8')
'multi_blob': 4 args specifying the maximum number of blobs, minimum and maximum number of iterations and threshold to generate a blob (default is '3,4,8,0.6')
'random_multi_blob': 3 args specifying the maximum number of blobs, minimum and maximum number of iterations to generate a blob (default is '3,4,8')
'fixed_random': 2 args specifying the minimum and maximum number of visible pixels (default is '10,20')
'random': 1 arg specifying the probability of a pixel being visible (default is '0.15')
'''

flags.DEFINE_string("im_shape", "32,32", "Comma-separated image shape [height, width] (default: '32,32')")
flags.DEFINE_integer("num_channels", 3, "Number of channels, options are (i)3 and (ii)1 (default: 3)")
flags.DEFINE_integer("num_colors", 256, "Number of colors a pixel can have, options are (i)256 and (i)2 (default: 256)")
flags.DEFINE_integer("buffer_size", 200000, "buffer_size for shuffling the data in train mode (default: 200000)")

# Other params 
flags.DEFINE_string("loss_func", 'heuristic', "Loss function to be used, options are (i) heuristic, (ii) weighted and (iii) vanilla_pcnn (default: heuristic)")
flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 50)")
flags.DEFINE_integer("sample_every", 1000, "Sample images after this many steps (default: 1000)")
flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 5000)")
flags.DEFINE_float("sample_temp", 1.1, "Temperature to be used for tempered softmax while sampling (default: 1.1)")

# Optimizer params
flags.DEFINE_integer("device_id", 0, "gpu device id")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")

conf = flags.FLAGS

def main(_):
  solver = Solver()
  solver.train()

if __name__ == '__main__':
  tf.app.run()
