from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import os, sys 
import time
sys.path.append('/models')
from official.benchmark.models import trivial_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import common
from official.vision.image_classification import imagenet_preprocessing

import horovod.tensorflow as hvd

flags.DEFINE_string(name='cnn_model', default='resnet101', help='model to test')

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


class TimeHistory(object):
  def __init__(self, batch_size, log_steps):
    self.batch_size = batch_size
    self.log_steps = log_steps ## typically the number of steps in each epoch
    self.global_steps = 0
    self.epoch_num = 0
    self.examples_per_second = 0
    logging.info("batch steps: %f", log_steps)

  def on_train_end(self):
    self.train_finish_time = time.time()
    elapsed_time = self.train_finish_time - self.train_start_time
    logging.info(
      "total time take: %f,"
      "averaged examples_per_second: %f",
      elapsed_time, self.examples_per_second / self.epoch_num)

  def on_epoch_begin(self, epoch):
    self.epoch_num += 1
    self.epoch_start = time.time()

  def on_batch_begin(self, batch):
    self.global_steps += 1
    if self.global_steps == 1:
      self.train_start_time = time.time()
      self.start_time = time.time()

  def on_batch_end(self, batch, loss):
    """Records elapse time of the batch and calculates examples per second."""
    logging.info(
      "global step:%d, loss value: %f",
      self.global_steps, loss)
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      logging.info(
        "global step:%d, time_taken: %f,"
        "examples_per_second: %f",
          self.global_steps, elapsed_time, examples_per_second)
      self.examples_per_second += examples_per_second
      self.start_time = timestamp

  def on_epoch_end(self, epoch):
    epoch_run_time = time.time() - self.epoch_start
    logging.info(
      "epoch':%d, 'time_taken': %f",
      epoch, epoch_run_time)

def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:/storage/baoyu/arion/scalability/models-master/official/vision/image_classification/resnet_imagenet_main.py
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
    policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
        'mixed_float16', loss_scale=loss_scale)
    tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)
    if not keras_utils.is_v2_0():
      raise ValueError('--dtype=fp16 is not supported in TensorFlow 1.')
  elif dtype == tf.bfloat16:
    policy = tf.compat.v1.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v1.keras.mixed_precision.experimental.set_policy(policy)

  data_format = flags_obj.data_format

  input_fn = imagenet_preprocessing.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  #if flags_obj.cnn_model == 'vgg16':
  if 'vgg' in flags_obj.cnn_model:
    lr_schedule = 0.01  
  else:
    lr_schedule = 0.1
  if flags_obj.use_tensor_lr:
    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)

  #with strategy_scope:

  with tf.Graph().as_default():

    train_input_dataset = input_fn(
        is_training=True,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        num_epochs=flags_obj.train_epochs,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        dtype=dtype,
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
        training_dataset_cache=flags_obj.training_dataset_cache,
    )

    # TODO(hongkuny): Remove trivial model usage and move it to benchmark.
    if flags_obj.cnn_model == 'resnet101':
      model = tf.keras.applications.ResNet101(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'vgg16':
      model = tf.keras.applications.VGG16(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'inceptionv3':
      model = tf.keras.applications.InceptionV3(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.cnn_model == 'densenet121':
      model = tf.keras.applications.DenseNet121(
        weights=None,
        classes=imagenet_preprocessing.NUM_CLASSES)
    else:
      raise ValueError('Other Model Undeveloped')


    optimizer = tf.keras.optimizers.Adam(
      learning_rate=lr_schedule,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-08)

    optimizer = hvd.DistributedOptimizer(optimizer)

    train_input_iterator = tf.compat.v1.data.make_one_shot_iterator(train_input_dataset)
    train_input, train_target = train_input_iterator.get_next()

    steps_per_epoch = (
        imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
    train_epochs = flags_obj.train_epochs

    # callbacks = common.get_callbacks(steps_per_epoch,
    #                                  common.learning_rate_schedule)
    if flags_obj.enable_checkpoint_and_export:
      ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
      # callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
      #                                                     save_weights_only=True))

    # if mutliple epochs, ignore the train_steps flag.
    if train_epochs <= 1 and flags_obj.train_steps:
      steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
      train_epochs = 1

    num_eval_steps = (
        imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)


    train_output = model(train_input, training=True)
    scc_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    #loss = tf.compat.v1.keras.losses.sparse_categorical_crossentropy(train_target,
    # train_output)
    loss = scc_loss(train_target, train_output)
    var_list = variables.trainable_variables() + \
      ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
    grad = optimizer.get_gradients(loss, var_list)
    train_op = optimizer.apply_gradients(zip(grad, var_list))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer()) 
    summary = TimeHistory(flags_obj.batch_size, steps_per_epoch)
    for epoch_id in range(train_epochs):
      summary.on_epoch_begin(epoch_id)
      for batch_id in range(steps_per_epoch):
        summary.on_batch_begin(batch_id)
        loss_v, _ = sess.run([loss, train_op])

        if batch_id == 0:
          hvd.broadcast_variables(model.variables, root_rank=0)
          hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        summary.on_batch_end(batch_id, loss_v)
      summary.on_epoch_end(epoch_id)
    summary.on_train_end()
  return

def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  logdir = './logs'
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  else:
    logname = 'log_{}'.format(flags.FLAGS.cnn_model)
  logging.get_absl_handler().use_absl_log_file(logname, logdir)
  with logger.benchmark_context(flags.FLAGS):
    run(flags.FLAGS)
  #logging.info('Run stats:\n%s', stats)
  with open('end.o', 'w') as f:
    f.write('this test is done')
  exit()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)
