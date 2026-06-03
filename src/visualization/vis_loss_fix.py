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

def find_event_file(event_dir):
    """Return the first TensorBoard event file found in log_dir."""
    try:
        files = os.listdir(event_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {event_dir}")
    
    event_files = [f for f in files if f.startswith("events.out.tfevents")]
    if not event_files:
        raise FileNotFoundError(f"No event file in {event_dir}")
    
    return os.path.join(event_dir, event_files[0])

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

def steps_to_epochs(step_counter, batches_per_epoch, d_loop=None):
    """
    Convert step counter to epoch fraction.
    For generator steps: epoch = step_counter / batches_per_epoch
    For critic steps: epoch = step_counter / (batches_per_epoch * d_loop)
    """
    if d_loop is None:
        # generator steps
        return np.array(step_counter) / batches_per_epoch
    else:
        # critic steps
        return np.array(step_counter) / (batches_per_epoch * d_loop)

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
    log_dir = find_event_file(event_dir=event_dir)

    print_available_tags(log_dir)

    steps_dis, losses_dis = read_tensorboard_logs(log_dir, "Loss/Critic")

    steps_fake, losses_fake = read_tensorboard_logs(log_dir, "Loss/D_fake")
    steps_real, losses_real = read_tensorboard_logs(log_dir, "Loss/D_real")

    steps_gen, losses_gen = read_tensorboard_logs(log_dir, "Loss/Generator")

    steps_gp, losses_gp = read_tensorboard_logs(log_dir, "GP")

    # After these lines:
    steps_gp, losses_gp = read_tensorboard_logs(log_dir, "GP")

    # ---------- INSERT CONVERSION CODE HERE ----------
    batch_size = parameters['batch_size']
    d_loop = parameters['d_loop']
    num_samples = parameters.get('num_samples', None)

    if num_samples is None:
        # fallback: estimate from max generator step and num_epochs
        max_gen_step = max(steps_gen)
        num_epochs = parameters['num_epochs']
        batches_per_epoch = max_gen_step / num_epochs
    else:
        batches_per_epoch = num_samples // batch_size

    # Convert steps to epochs
    epochs_gen = np.array(steps_gen) / batches_per_epoch
    epochs_critic = np.array(steps_dis) / (batches_per_epoch * d_loop)
    epochs_fake = np.array(steps_fake) / (batches_per_epoch * d_loop)
    epochs_real = np.array(steps_real) / (batches_per_epoch * d_loop)
    epochs_gp = np.array(steps_gp) / (batches_per_epoch * d_loop)
    # ---------- END OF INSERTION ----------

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the 2D array of axes to make indexing easier (0, 1, 2)
    axs_flat = axs.flatten()

    # Subplot 0
    axs_flat[0].plot(epochs_gen, losses_gen, label="Gen Loss")
    axs_flat[0].plot(epochs_gen, moving_avg(losses_gen)[9:-9], linestyle='--', label='Centered MA')
    axs_flat[0].hlines(y=0, xmin=epochs_gen[0]-0.5, xmax=epochs_gen[-1]+0.5, linestyle='--', color='r', alpha=0.6)
    axs_flat[0].set_xlim(epochs_gen[0]-0.5, epochs_gen[-1]+0.5)
    axs_flat[0].set_xlabel('Epoch')
    axs_flat[0].set_ylabel('Loss Score')
    axs_flat[0].legend()
    axs_flat[0].set_title('Generator Loss')

    # Subplot 1
    axs_flat[1].plot(epochs_critic, losses_dis, label='Critic Loss')
    axs_flat[1].plot(epochs_critic, moving_avg(losses_dis)[9:-9], linestyle='--', label='Centered MA')
    axs_flat[1].hlines(y=0, xmin=epochs_critic[0]-0.5, xmax=epochs_critic[-1]+0.5, linestyle='--', color='r', alpha=0.6)
    axs_flat[1].set_xlim(epochs_critic[0]-0.5, epochs_critic[-1]+0.5)
    axs_flat[1].set_xlabel('Epoch')
    axs_flat[1].set_ylabel('Loss Score')
    axs_flat[1].legend()
    axs_flat[1].set_title('Critic Loss')

    # Subplot 2
    axs_flat[2].plot(epochs_gp, losses_gp, label='GP')
    axs_flat[2].hlines(y=0, xmin=epochs_gp[0]-0.5, xmax=epochs_gp[-1]+0.5, linestyle='--', color='r', alpha=0.6)
    axs_flat[2].set_xlim(epochs_gp[0]-0.5, epochs_gp[-1]+0.5)
    axs_flat[2].set_xlabel('Epoch')
    axs_flat[2].set_ylabel('Penalty Norm')
    axs_flat[2].set_title('Gradient Penalty')

    # Subplot 3
    axs_flat[3].plot(epochs_fake, losses_fake, label='Fake samples')
    axs_flat[3].plot(epochs_real, losses_real, label='Real samples')
    axs_flat[3].fill_between(epochs_fake, losses_fake, losses_real, color='gray', alpha=0.2, label='Distinction Gap')
    axs_flat[3].hlines(y=0, xmin=epochs_fake[0]-0.5, xmax=epochs_fake[-1]+0.5, linestyle='--', color='r', alpha=0.6)
    axs_flat[3].set_xlim(epochs_fake[0]-0.5, epochs_fake[-1]+0.5)
    axs_flat[3].set_xlabel('Epoch')
    axs_flat[3].set_ylabel('Loss Score')
    axs_flat[3].legend()
    axs_flat[3].set_title('Real vs Fake Loss')

    # Adjust the spacing between subplots 
    plt.tight_layout()

    # Save img
    path = os.path.join(image_dir, 'loss_curves.png')
    plt.savefig(path, format="PNG")

    plt.show()

if __name__ == "__main__":
    main()