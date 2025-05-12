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

# Create a callback to track loss values
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

class LossTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossTrackingCallback, self).__init__(verbose)
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.iterations = []
        self.iter_count = 0
        
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # Extract loss values from logger
        if len(self.model.logger.name_to_value) > 0:
            self.iter_count += 1
            self.iterations.append(self.iter_count)
            
            # Extract different loss components if available
            if 'train/loss' in self.model.logger.name_to_value:
                self.losses.append(self.model.logger.name_to_value['train/loss'])
            if 'train/policy_gradient_loss' in self.model.logger.name_to_value:
                self.policy_losses.append(self.model.logger.name_to_value['train/policy_gradient_loss'])
            if 'train/value_loss' in self.model.logger.name_to_value:
                self.value_losses.append(self.model.logger.name_to_value['train/value_loss'])
            if 'train/entropy_loss' in self.model.logger.name_to_value:
                self.entropy_losses.append(self.model.logger.name_to_value['train/entropy_loss'])
        return True

# Create the callback
loss_callback = LossTrackingCallback()

# Train the agent with the callback
model.learn(total_timesteps=100000, callback=loss_callback)

# Save the model
model.save("ppo_chopsticks_model")

# Plot the loss values
plt.figure(figsize=(12, 8))

# Plot overall loss
if len(loss_callback.losses) > 0:
    plt.subplot(2, 2, 1)
    plt.plot(loss_callback.iterations, loss_callback.losses)
    plt.title('Overall Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

# Plot policy gradient loss
if len(loss_callback.policy_losses) > 0:
    plt.subplot(2, 2, 2)
    plt.plot(loss_callback.iterations, loss_callback.policy_losses)
    plt.title('Policy Gradient Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

# Plot value loss
if len(loss_callback.value_losses) > 0:
    plt.subplot(2, 2, 3)
    plt.plot(loss_callback.iterations, loss_callback.value_losses)
    plt.title('Value Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')  # Use log scale for value loss as it can be very small

# Plot entropy loss
if len(loss_callback.entropy_losses) > 0:
    plt.subplot(2, 2, 4)
    plt.plot(loss_callback.iterations, loss_callback.entropy_losses)
    plt.title('Entropy Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)

plt.tight_layout()
plt.savefig('training_losses.png')
plt.show()
# %%
