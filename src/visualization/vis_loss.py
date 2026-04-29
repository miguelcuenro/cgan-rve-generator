# Import required 
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import torch
import os
import sys
import yaml

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -------------------- #
# FUNCTION DEFINITIONS
# -------------------- #

def read_tensorboard_logs(log_path, tag):
    # Initialize the event accumulator
    event_acc = EventAccumulator(log_path)
    event_acc.Reload() # Load data from the files

    # Extract a specific tag (e.g., 'Loss/Critic') 
    events = event_acc.Scalars(tag)
    
    # Store steps and values
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values

def print_available_tags(log_path):
    # Initialize and load the accumulator
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    
    # Get all tags
    tags = event_acc.Tags()
    
    print(f"Available tags in: {log_path}\n" + "-"*40)
    
    # Iterate through the categories (scalars, histograms, images, etc.)
    for category, tag_list in tags.items():
        if tag_list: # Only print if there are tags in this category
            print(f"[{category.upper()}]:")
            for tag in tag_list:
                print(f"  - {tag}")
    print("-"*40)

def moving_avg(data, window_size=10):
    if isinstance(data, list):
        data = np.array(data)

    pad_data = np.pad(data, (window_size-1, window_size-1), mode='edge')
    ma = np.convolve(pad_data, np.ones(window_size), mode='same') / window_size

    return ma

# -------------- #
# MAIN EXECUTION
# -------------- #

def main():
    # Load the parameters file
    with open('parameters.yaml', 'r') as dictionary:
        parameters = yaml.safe_load(dictionary)

    # Raise an error if parameters.yaml is NOT a dictionary
    if not isinstance(parameters, dict):
        raise TypeError("parameters.yaml did not parse to a dictionary. Please check the file's structure.")

    event_dir =os.path.expanduser(parameters['event_dir'])
    dataroot = os.path.expanduser(parameters['dataroot'])
    image_dir = os.path.join(dataroot, "img")

    # Guidelines
    print("Generator Loss:")
    print("Expect a Plateau. You aren't looking for this to go to zero. Once it stabilizes, it means the Generator has found a consistent way to fool the Critic.")
    print(15*'-')
    print('Critic Loss:')
    print('Expect Stable Negative Values. Since the Critic is minimizing the negative Wasserstein distance, it should hover in a stable negative range. Constant diving into deep negative numbers indicates instability.')
    print(15*'-')
    print('Gradient Penalty (GP):')
    print('Expect Near-Zero. This is your safety rail. If this value starts climbing rapidly, your training is becoming "unconstrained," and your gradients are likely about to explode.')
    print(15*'-')
    print('Real vs. Fake Gap:')
    print('Expect a consistent, positive gap; the Real samples should remain above the Fake samples, and as training progresses, you want the Fake line to trend upward toward the Real line until the distance stabilizes at a low value.')

    # Usage
    log_dir = event_dir + "/events.out.tfevents.1775907087.MacBook-Pro-de-Miguel.local.8267.0" 

    print_available_tags(log_dir)

    steps_dis, losses_dis = read_tensorboard_logs(log_dir, "Loss/Critic")

    steps_fake, losses_fake = read_tensorboard_logs(log_dir, "Loss/D_fake")
    steps_real, losses_real = read_tensorboard_logs(log_dir, "Loss/D_real")

    steps_gen, losses_gen = read_tensorboard_logs(log_dir, "Loss/Generator")

    steps_gp, losses_gp = read_tensorboard_logs(log_dir, "GP")

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the 2D array of axes to make indexing easier (0, 1, 2)
    axs_flat = axs.flatten()

    # Plotting the three subplots
    axs_flat[0].plot(steps_gen, losses_gen, label="Gen Loss")
    axs_flat[0].plot(steps_gen, moving_avg(data=losses_gen, window_size=10)[9:-9], linestyle='--', label='Centered MA')
    axs_flat[0].hlines(y=0, xmin=steps_gen[0]-1e2, xmax=steps_gen[-1]+1e2, linestyle='--', color='r', alpha=0.6)

    axs_flat[0].set_xlabel('Iteration')
    axs_flat[0].set_ylabel('Loss Score')
    axs_flat[0].set_xlim(-5, steps_gen[-1]+5)

    axs_flat[0].legend()
    axs_flat[0].set_title('Subplot 1: Generator Loss')

    axs_flat[1].plot(steps_dis, losses_dis, label='Critic Loss')
    axs_flat[1].plot(steps_dis, moving_avg(data=losses_dis, window_size=10)[9:-9], linestyle='--', label='Centered MA')
    axs_flat[1].hlines(y=0, xmin=steps_dis[0]-1e2, xmax=steps_dis[-1]+1e2, linestyle='--', color='r', alpha=0.6)

    axs_flat[1].set_xlabel('Iteration')
    axs_flat[1].set_ylabel('Loss Score')
    axs_flat[1].set_xlim(-5, steps_dis[-1]+5)

    axs_flat[1].legend()
    axs_flat[1].set_title('Subplot 2: Critic Loss')

    axs_flat[2].plot(steps_gp, losses_gp, label='')
    axs_flat[2].hlines(y=0, xmin=steps_gp[0]-1e2, xmax=steps_gp[-1]+1e2, linestyle='--', color='r', alpha=0.6)

    axs_flat[2].set_xlabel('Iteration')
    axs_flat[2].set_ylabel('Penalty Norm')
    axs_flat[2].set_xlim(-5, steps_gp[-1]+5)

    axs_flat[2].set_title('Subplot 3: Gradient Pentalty')

    wd = np.array(losses_fake)-np.array(losses_real)

    axs_flat[3].plot(steps_fake, losses_fake, label='Fake samples')
    axs_flat[3].plot(steps_real, losses_real, label='Real samples')
    # axs_flat[3].plot(steps_real, wd, color='grey', label='1-Wassertein Dist.')
    axs_flat[3].fill_between(
        steps_fake,           # X-axis
        losses_fake,          # Upper boundary
        losses_real,          # Lower boundary
        color='gray',         # Color of the shading
        alpha=0.2,            # Transparency (0.0 to 1.0)
        label='Distinction Gap'
    )
    axs_flat[3].hlines( y=0, xmin=steps_gp[0]-1e2, xmax=steps_gp[-1]+1e2, linestyle='--', color='r', alpha=0.6)

    axs_flat[3].set_xlabel('Iteration')
    axs_flat[3].set_ylabel('Loss Score')
    axs_flat[3].set_xlim(-5, steps_gp[-1]+5)

    axs_flat[3].legend()
    axs_flat[3].set_title('Subplot 4: Real Sample Loss vs Fake Sample Loss')

    plt.tight_layout()
    plt.show()

    # Save img
    path = os.path.join(image_dir, 'loss_curves.png')
    plt.savefig(path, format="PNG")

if __name__ == "__main__":
    main()