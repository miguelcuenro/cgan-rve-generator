import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import os
import sys
import datetime

def visualize_tensor(tensor, colormap='gist_earth', show_on_screen=True):
    """
    Visualizes a 3D tensor using PyVista and optionally displays the plot on-screen.

    :param tensor: 3D NumPy array of shape [64, 64, 64].
    :param colormap: Colormap for the visualization.
    :param show_on_screen: If True, displays the plot on-screen. If False, off-screen rendering is used.

    # Create a PyVista grid object from the tensor
    grid = pv.ImageData()
    grid.dimensions = np.array(tensor.shape) + 1
    grid.spacing = (1, 1, 1)  # Assuming unit spacing, adjust as necessary
    grid.cell_data["values"] = tensor.flatten(order="F")

    # Setup the camera position
    camera_position = [
        (200, 200, 200),  # Camera location
        (32, 32, 32),  # Focal point
        (0, 0, 1)  # View up direction
    ]

    # Configure the plotter
    plotter = pv.Plotter(off_screen=False)  # off_screen=not show_on_screen)
    plotter.add_volume(grid, cmap=colormap)
    plotter.camera_position = camera_position

    if show_on_screen:
        # Display the plot interactively
        plotter.show()
    else:
        # Or use off-screen rendering to capture the plot as an image
        img = plotter.screenshot()
        return img  # You can then use this image to log to TensorBoard
    """
    # Ensure tensor is a NumPy array
    if isinstance(tensor, torch.Tensor):
        # tensor = tensor.cpu().numpy()
        tensor = tensor.detach().numpy()

    if tensor.ndim == 4 and tensor.shape[0] == 1:  # Check for single-channel 4D tensor
        tensor = tensor.squeeze(0)  # Remove channel dimension, resulting in 3D

    assert tensor.ndim == 3, "Tensor must be 3D after squeezing."

    if colormap == 'rand':
        colors = np.random.rand(1024, 4)
        for i in range(0, colors.shape[0], 1):
            colors[i][3] = 1
        colormap = ListedColormap(colors)

    # Convert the 3D tensor to a PyVista UniformGrid
    grid = pv.ImageData(dimensions=(tensor.shape[0] + 1, tensor.shape[1] + 1, tensor.shape[2] + 1),
                        spacing=(1, 1, 1))
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = tensor.flatten(order='F')  # Flatten the tensor in Fortran order

    # Render the volume using PyVista
    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(grid, cmap=colormap)
    #plotter.show(full_screen=False)
    return plotter

# Load the parameters file
with open('parameters.yaml', 'r') as dictionary:
    parameters = yaml.safe_load(dictionary)

sampling_dir = parameters['sampling_dir']

for filename in os.listdir(sampling_dir):
    npy_file = np.load(os.path.join(sampling_dir, filename))
    tensor = torch.from_numpy(npy_file)
    visualize_tensor(tensor).save_graphic(sampling_dir + f'/rve_plot_{filename.split('_')[1].split('.')[0]}.pdf')