from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from tf_utils import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_integer('eval_steps', 1000, 'Number of steps to run trainer.')
tf.app.flags.DEFINE_string('train_dir', 'train', 'Directory to put the training data.')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('fake_data', False, 'Use fake data.  ')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(x, keep_prob):
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv

def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', cross_entropy)
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate=1e-4):
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def _mnist_fs(x, y_, keep_prob):
  y = inference(x, keep_prob)
  cross_entropy = loss(y, y_)
  tf.add_to_collection("losses", cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y,1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return {"inference":y, "loss":cross_entropy, "accuracy":accuracy, "losses":[]}

def mnist_fs():
  x = tf.placeholder(tf.float32, [None, 784])
  keep_prob = tf.placeholder(tf.float32)
  y_ = tf.placeholder(tf.int64, [None]) #,10
  pl_dict = {'x':x, 'y':y_, 'keep_prob':keep_prob}
  return pl_dict, _mnist_fs(x, y_, keep_prob)

class BatchFeederD:
  def __init__(self, data):
    self.data = data
    self.num_examples = data.num_examples
  def next_batch(self, batch_size, *args):
    (x, y) = self.data.next_batch(batch_size, *args)
    return {"x": x, "y": y}

def main(_):
  data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
  train_data = BatchFeederD(data_sets.train)
  train_data_copy = BatchFeederD(data_sets.train)
  test_data = BatchFeederD(data_sets.test)
  addons = [GlobalStep(),
                TrackAverages(), #do this before train (why?)
                Train(lambda gs: tf.train.AdamOptimizer(1e-4), FLAGS.batch_size, train_feed={'keep_prob' : 0.5}, print_steps=100),
                Histograms(), #includes gradients, so has to be done after train
                Saver(save_steps = 1000, checkpoint_path = 'model.ckpt'),
                #SummaryWriter(summary_steps = 100, feed_dict = {'keep_prob': 1.0}),
                Eval(train_data_copy, FLAGS.batch_size, ['accuracy'], eval_feed={'keep_prob': 1.0}, eval_steps = 1000, name="training"),
                Eval(test_data, FLAGS.batch_size, ['accuracy'], eval_feed={'keep_prob': 1.0}, eval_steps = 1000, name="test")]
  pl_dict, model = mnist_fs()
  trainer = Trainer(model, FLAGS.max_steps, train_data, addons, pl_dict, train_dir = "train/", verbosity=1)
  trainer.init_and_train()

if __name__=="__main__":  
  tf.app.run()

