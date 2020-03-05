import cv2
import gym
import random
import numpy as np

class Environment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name)
    print (config.action_repeat)
    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._previous_screen = None
    self._screen = None
    self.reward = 0
    self.terminal = True
    self.minus_one_if_dead = config.minus_one_if_dead
    self.initialized = False

  def new_game(self, from_random_game=False):
    if self.lives == 0 or not self.initialized:
      self._screen = self.env.reset()
      self.initialized = True
    self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

  def new_random_game(self):
    self.new_game(True)
    for _ in range(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

  def _step(self, action):
    self._previous_screen = self._screen
    obs, self.reward, self.terminal, _ = self.env.step(action)
    self._screen = obs
    self._observation = self._screen

  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @ property
  def screen(self):
    return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    return self.env.ale.lives()

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()

  def after_act(self, action):
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in range(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      if is_training and start_lives > self.lives:
        if self.minus_one_if_dead:
          cumulated_reward += -1
        self.terminal = True

      if self.terminal:
        break

    self.reward = cumulated_reward
    self._screen = np.maximum(self._screen, self._previous_screen)

    self.after_act(action)
    return self.state

class SimpleGymEnvironment(Environment):
  def __init__(self, config):
    super(SimpleGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action)

    self.after_act(action)
    return self.state
