#%%
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.simulation_env import SimulationEnv
from utils.gym_wrapper import GymWrapper

#%%

# Initialize and wrap the environment
xml_path = "universe.xml"  # Replace with actual path
custom_env = SimulationEnv(xml_path)
wrapped_env = DummyVecEnv([lambda: GymWrapper(custom_env)])

# Optional: check environment compliance
check_env(wrapped_env.envs[0], warn=True)

#%%
# Initialize PPO model
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_chopsticks_model")
# %%
