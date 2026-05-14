# Import required 
import numpy as np
import os
import sys
import pyvista as pv
import yaml
from natsort import natsorted

sys.path.append(os.path.abspath('../utils'))
import visualization_utils as vu

from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import matplotlib.pyplot as plt

# -------------- #
# MAIN EXECUTION
# -------------- #

def main():
    pv.global_theme.jupyter_backend = 'static'  # Best for screenshots
    print("PyVista ready (static PNGs)")

    # Load the parameters file
    with open('parameters.yaml', 'r') as dictionary:
        parameters = yaml.safe_load(dictionary)

    # Raise an error if parameters.yaml is NOT a dictionary
    if not isinstance(parameters, dict):
        raise TypeError("parameters.yaml did not parse to a dictionary. Please check the file's structure.")

    dataroot = os.path.expanduser(parameters['dataroot'])
    image_dir = os.path.join(dataroot, "img")

    # Make the dir to store the images if non existent
    os.makedirs(image_dir, exist_ok=True)

    # Visualize samples
    labels = np.load(os.path.join(dataroot, 'labels.npy'))
    cnt = 0

    files = (file for file in os.listdir(dataroot) if os.path.isfile(os.path.join(dataroot, file)))
    sorted_files = natsorted(files)

    print("-"*45)
    print("Rendering RVEs")
    print("-"*45)
        
    for file in sorted_files:
        if file != "labels.npy":
            print(file)
            rve = np.load(os.path.join(dataroot, file))
            img = vu.visualize_tensor(rve)  # Returns PNG array!

            # Plot img
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.axis('off')

            # Add title
            title = f"Ferrite phase ratio $l = {round(labels[cnt]*1e-2, 3)}$"
            plt.title(title)

            # Save img
            path = os.path.join(image_dir, 'rve_'+str(cnt)+'.png')
            plt.savefig(path, format="PNG")

            # Close img to save up memory
            plt.close()

            # Increase counter
            cnt += 1

    print("-"*45)
    print("Render complete!")
    print("Look at them in", image_dir)
    print("-"*45)

if __name__ == "__main__":
    main()