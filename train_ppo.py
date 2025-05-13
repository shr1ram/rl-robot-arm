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
custom_env = SimulationEnv(xml_path, max_steps=20000)
wrapped_env = DummyVecEnv([lambda: GymWrapper(custom_env)])

# Optional: check environment compliance
check_env(wrapped_env.envs[0], warn=True)

#%%
# Initialize PPO model
# Setting n_steps parameter to control batch size per update
n_steps = 2048  # Default PPO batch size

# Increase entropy coefficient to encourage exploration
ent_coef = 0.01  # Default is 0.0

# Use a larger network for more capacity
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
)

model = PPO(
    "MlpPolicy", 
    wrapped_env, 
    verbose=1, 
    n_steps=n_steps,
    ent_coef=ent_coef,  # Higher entropy encourages exploration
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # GAE parameter
    clip_range=0.2  # PPO clip parameter
)

# Create a callback to track loss values and save models periodically
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import os

class LossTrackingCallback(BaseCallback):
    def __init__(self, verbose=0, save_freq=25, save_path="./models"):
        super(LossTrackingCallback, self).__init__(verbose)
        self.losses = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.iterations = []
        self.iter_count = 0
        
        # Parameters for periodic saving
        self.save_freq = save_freq  # Save every N iterations
        self.save_path = save_path
        
        # Create models directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # Get the loss values from the model's logger
        if self.model.logger is not None and hasattr(self.model.logger, "name_to_value"):
            # Collect loss values
            policy_loss = None
            value_loss = None
            entropy_loss = None
            total_loss = None
            
            for key, value in self.model.logger.name_to_value.items():
                if key == "train/policy_gradient_loss":
                    policy_loss = value
                elif key == "train/value_loss":
                    value_loss = value
                elif key == "train/entropy_loss":
                    entropy_loss = value
                elif key == "train/loss":
                    total_loss = value
            
            # Only increment iteration counter and append values if we have data
            if any([policy_loss, value_loss, entropy_loss, total_loss]):
                # Increment iteration counter
                self.iter_count += 1
                self.iterations.append(self.iter_count)
                
                # Append loss values
                if total_loss is not None:
                    self.losses.append(total_loss)
                if policy_loss is not None:
                    self.policy_losses.append(policy_loss)
                if value_loss is not None:
                    self.value_losses.append(value_loss)
                if entropy_loss is not None:
                    self.entropy_losses.append(entropy_loss)
                
                # Save the model periodically
                if self.iter_count % self.save_freq == 0:
                    save_path = os.path.join(self.save_path, f"ppo_chopsticks_model_{self.iter_count}")
                    self.model.save(save_path)
                    print(f"\nModel saved at iteration {self.iter_count}: {save_path}\n")
        
        return True

# Create the callback with periodic saving every 25 iterations
loss_callback = LossTrackingCallback(save_freq=25, save_path="./models")

# Train the agent with the callback
# Calculate total_timesteps based on desired number of iterations
iterations = 50
total_timesteps = n_steps * iterations
print(f"Training for {iterations} iterations ({total_timesteps} timesteps)")
model.learn(total_timesteps=total_timesteps, callback=loss_callback)

# Save the final model
final_model_path = "./models/ppo_chopsticks_model_final"
model.save(final_model_path)
print(f"Final model saved at: {final_model_path}")

# Also save a copy with a simple name for easy loading
model.save("ppo_chopsticks_model")
print("Also saved as: ppo_chopsticks_model (for backward compatibility)")


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
