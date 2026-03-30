import numpy as np
import os
import sys
import pyvista as pv
import yaml

from natsort import natsorted

sys.path.append(os.path.abspath('../utils'))
import visualization_utils as vu

import matplotlib.pyplot as plt

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

# Load lables and sample RVEs
labels = np.load(os.path.join(dataroot, 'labels.npy'))
pr_list = []

files = (file for file in os.listdir(dataroot) if os.path.isfile(os.path.join(dataroot, file)))
sorted_files = natsorted(files)

for file in sorted_files:
    if file != "labels.npy":
        rve = np.load(os.path.join(dataroot, file))
        ferrite_phase_ratio = 1 - np.mean(rve)
        pr_list.append(ferrite_phase_ratio)

pr_np = np.array(pr_list)

print("Labels")
print(labels)
print("np.mean()")
print(pr_np)

# Compute metrics

error = pr_np - labels
mean_error = np.mean(error)
sme = np.mean(error**2)
rsme = np.sqrt(sme)
std = np.std(error)

# Visualize phase ratio
x = np.arange(0, 16)

plt.figure(figsize=(5,5))

plt.plot(x, pr_np, label="Sample phase ratios")
plt.plot(x, labels * 1e-2, label="Target phase ratios")

plt.xlabel("Sample $n$")
plt.ylabel("Phase ratio")

plt.ylim(-0.1, 1.1)

plt.legend()

plt.title("Sample vs Label Phase Ratios")

# Save img
path = os.path.join(image_dir, 'phase_ratios.png')
plt.savefig(path, format="PNG")

# Visualize statistics

plt.figure(figsize=(5,5))

plt.scatter(labels, error, alpha=0.6, s=50, label='Samples')

# Zero line
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Perfect')

# Mean error line with std band
plt.axhline(y=mean_error, color='b', linestyle='-', linewidth=2, label=f'Mean error: {mean_error:.3f}')
plt.fill_between([min(labels), max(labels)], 
                 mean_error - std, 
                 mean_error + std, 
                 alpha=0.2, color='blue', 
                 label=f'±1 STD: {std:.3f}')

# Save img
path = os.path.join(image_dir, 'errors.png')
plt.savefig(path, format="PNG")
