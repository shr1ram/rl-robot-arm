import gymnasium as gym
import numpy as np
import argparse
import os
from stable_baselines3 import PPO
from utils.simulation_env import SimulationEnv
from utils.gym_wrapper import GymWrapper
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize a trained policy')
parser.add_argument('--model', type=str, default="ppo_chopsticks_model",
                    help='Model path to load (default: ppo_chopsticks_model)')
parser.add_argument('--iteration', type=str, default=None,
                    help='Iteration number to load (will look in ./models/ppo_chopsticks_model_X)')
args = parser.parse_args()

# Initialize the environment
xml_path = "universe.xml"  # Using the universe model that contains chopsticks and cube
custom_env = SimulationEnv(xml_path)
env = GymWrapper(custom_env)

# Determine which model to load
if args.iteration is not None:
    model_path = f"./models/ppo_chopsticks_model_{args.iteration}"
    if not os.path.exists(model_path + ".zip"):
        print(f"Warning: Model at iteration {args.iteration} not found. Using default model.")
        model_path = args.model
else:
    model_path = args.model

# Load the trained model
print(f"Loading model: {model_path}")
model = PPO.load(model_path)

# Run the simulation with visualization
obs, _ = env.reset()
print("Starting simulation with the trained policy...")
print("The policy should pick up the cube and raise it to maximum height")
print("Press Ctrl+C to stop the simulation")
print("NOTE: This script must be run with 'mjpython' on macOS")

# Set the maximum number of steps for a single run
MAX_STEPS = 1000

try:
    # Run the simulation once
    step_count = 0
    done = False
    
    print("\nStarting simulation (single run)")
    
    # Run the simulation for up to MAX_STEPS
    while not done and step_count < MAX_STEPS:
        # Get action from policy with deterministic=True for better performance
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, _, info = env.step(action)
        
        # Render the simulation
        custom_env.render()
        
        # Print current cube height for monitoring
        cube_height = custom_env._get_cube_z_position() - custom_env.initial_cube_z
        
        # Print status every 10 steps for cleaner output
        # if step_count % 10 == 0 or step_count == MAX_STEPS - 1:
        #     print(f"Step {step_count}/{MAX_STEPS}, Cube height: {cube_height:.4f}")
        
        # Add a small delay to make the visualization smoother
        time.sleep(0.01)
        
        step_count += 1
    
    if done:
        print(f"\nSimulation complete - Maximum height reached!")
    else:
        print(f"\nSimulation complete - Maximum steps reached.")
            
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
    if hasattr(custom_env, '_viewer'):
        custom_env._viewer.close()