
import numpy as np
import pyvista as pv
import torchy

from matplotlib.colors import ListedColormap
from torch.utils.tensorboard import SummaryWriter

# Straight up from Xavi's code
def visualize_and_log_to_tensorboard(tag, tensor, step, writer, colormap='gist_earth'):
    """
    Visualizes a single-channel 3D tensor and logs the visualization as an image to TensorBoard.

    :param tag: Tag for saving in TensorBoard.
    :param tensor: 4D NumPy array of shape [1, D, H, W] where D, H, W are the depth, height, and width.
    :param step: Current step or epoch to log this image at.
    :param writer: An instance of SummaryWriter from TensorBoard.
    :param colormap: Colormap for the visualization.
    """


    # Ensure tensor is a NumPy array
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

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
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid, cmap=colormap)
    img = plotter.screenshot()
    plotter.close()

    # Convert the screenshot (HWC numpy array) to CHW format for TensorBoard
    img_tensor = np.transpose(img, (2, 0, 1))
    # img_tensor = torch.tensor(img_tensor, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
    writer.add_image(tag, img_tensor, global_step=step, dataformats='CHW')