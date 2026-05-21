import numpy as np
import os

from natsort import natsorted

def func_a():
    root = "../models/rve/test"

    rve = np.zeros((1, 32, 32, 32))
    rve[0, 16:, 16:, 16:] = 1
    sample = np.load(root+"/number_11.npy")

    print(np.mean(rve))

    for i in range(16):
        np.save(root + "/number_" + str(i) + ".npy", rve)
        sample = np.load(root+"/number_"+str(i)+".npy")

def func_b():
    root_dir = os.path.expanduser("/Users/miguelcuenca/Documents/Master/HiWi/cGAN/data/processed_data")
    output_dir = "../models/rve/test2"

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subdirs = natsorted(subdirs)
    selected = subdirs[:16]

    print(f"Found {len(subdirs)} directories. Processing first {len(selected)}:")

    ferrite_vals = []

    for i, subdir in enumerate(selected):
        phase_path = os.path.join(root_dir, subdir, "phase_grid.npy")
        label_path = os.path.join(root_dir, subdir, "label.npy")
        
        # Load the phase grid
        phase_grid = np.load(phase_path)
        label_data = np.load(label_path)

        first_element = label_data[0]  # Ferrite percentage
        ferrite_vals.append(first_element)
        
        # Save as rve_number_i.npy
        output_path = os.path.join(output_dir, f"number_{i}.npy")
        np.save(output_path, phase_grid)
        print(f"Saved: {output_path}")
    
    labels = np.array(ferrite_vals)
    print(labels)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    print("Done.")

l1 = np.load("../models/rve/test/labels.npy")
l2 = np.load("../models/rve/test2/labels.npy")
l3 = np.load("../models/rve/2026-05-13_11-13-51/labels.npy")

print(l1)
print("-"*45)
print(l2)
print("-"*45)
print(l3)