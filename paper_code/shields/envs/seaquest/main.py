# import cv2
import random
import tensorflow as tf

from dqn.seaquest_shielded_agent import ShieldedAgent
from dqn.environment import GymEnvironment, SimpleGymEnvironment
from config import get_config
from dqn.agent import Agent

flags = tf.app.flags
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_boolean('save_weight', False, 'Save weight from pickle file')
flags.DEFINE_boolean('load_weight', False, 'Load weight from pickle file')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('minus_one_if_dead', False, 'Whether to -1 to reward if a life is discounted')
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_integer('action_repeat', 1, 'The number of action to be repeated')
flags.DEFINE_boolean('shield_active', False, 'Whether a shield should be used')
flags.DEFINE_boolean('negative_reward', False, 'Whether the shield ')
FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
  raise ValueError("--gpu_fraction should be defined")

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print (" [*] GPU : %.4f" % fraction)
  return fraction

def main(_):
  gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    config = get_config(FLAGS) or FLAGS

    if config.env_type == 'simple':
      env = SimpleGymEnvironment(config)
    else:
      env = GymEnvironment(config)

    if not FLAGS.use_gpu:
      config.cnn_format = 'NHWC'

    if config.shield_active:
      agent = ShieldedAgent(config, env, sess)
    else:
      agent = Agent(config, env, sess)

    if FLAGS.is_train:
      agent.train()
    else:
      agent.play()

if __name__ == '__main__':
  tf.app.run()
