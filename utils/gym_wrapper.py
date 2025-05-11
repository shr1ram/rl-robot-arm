import gymnasium as gym
import numpy as np


class GymWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.env.model.nu,), dtype=np.float32)
        obs = self.env.reset()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.astype(np.float32), reward, bool(done), False, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset().astype(np.float32)
        return obs, {}