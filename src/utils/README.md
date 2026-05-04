# Utils

## `visualization_utils.py`

`visualization_utils.py` provides two functions for visualizing 3D volumetric data (e.g., RVEs, generated microstructures) using PyVista. This code was not written by me, but by my predecessor (KUDOS to Xavi!!).

In the following, I will explain what every function does and write down comments worthy of remembering:

#### `visualize_and_log_to_tensorboard(tag, tensor, step, writer, colormap='gist_earth')`

1. **Input conversion** – Converts PyTorch tensors to NumPy arrays using `.cpu().numpy()`. If the input is already a NumPy array, it proceeds directly.

2. **Dimension squeezing** – Removes a leading dimension of size 1. After that, it asserts the tensor is exactly 3‑dimensional.

3. **Colormap handling** – If `colormap == 'rand'`, it creates a random RGBA colormap with 1024 colors (alpha channel set to 1). Otherwise, it uses the provided colormap name (`default 'gist_earth'`).

4. **PyVista grid creation** – Builds a `pv.ImageData` grid with dimensions `(D+1, H+1, W+1)` and unit spacing. The tensor values are assigned to cell data after flattening in Fortran (column‑major) order `(order='F')`. This ordering is essential because PyVista expects the fastest‑changing index to be the X dimension.

5. **Rendering** – Uses an off‑screen Plotter, adds the mesh with the chosen colormap, and captures a screenshot.

6. **TensorBoard logging** – Transposes the screenshot from HWC `(height, width, channel)` to CHW `(channel, height, width)` and calls `writer.add_image` with `dataformats='CHW'`.

*Note: The line that normalizes the image to [0,1] is commented out; the raw uint8 values (0–255) are logged, which TensorBoard accepts.*

#### `visualize_tensor(tensor, colormap='gist_earth', show_on_screen=True)`

1. **Input conversion and squashing** – Same as above: torch → numpy, squeeze leading 1‑dimension if present, assert 3D.

2. **Colormap** – Identical logic as the first function (including the 'rand' option).

3. **PyVista grid** – Exactly the same grid construction with Fortran‑order flattening.

4. **Rendering** – Creates an off‑screen Plotter (note: the `show_on_screen` parameter is currently ignored; the function always returns a screenshot). After adding the mesh and capturing the image, the plotter is closed.

5. **Return value** – A NumPy array of shape `(height, width, 3)` (RGB) ready to be saved, displayed with matplotlib, or passed to other functions.

#### Important notes

* *Both functions rely on Fortran flattening – this is crucial for correct spatial alignment.*
* *The visualize_tensor function does not actually use show_on_screen; to enable interactive viewing, the Plotter should be instantiated with `off_screen=False` and `plotter.show()` called instead of `.screenshot()`.*
* *The random colormap `('rand')` does not fix a random seed, so multiple runs may produce different color mappings.*
* *The code includes a commented‑out alternative using `add_volume` for volume rendering; the active implementation uses `add_mesh`, which treats the scalar field as a surface‑colored mesh.*