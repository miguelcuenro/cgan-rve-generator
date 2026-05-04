# Visualization

Here the code to visualize, by the cGAN, generated samples. Depending on what you want to visualize, run `vis_rve.py` or `vis_loss.py`. There is a third script, `vis_phase_ratios.py`. However, that is a WIP as there is an interesting feature (a bug...) that causes the martensite and ferrite ratios to flip around, and unfortunately I am still figuring why.

## Parameter file

As for now (3rd of May 2026), there are only two variables in `parameters.yaml`: `dataroot` and `event_dir`.

* `dataroot`: Path to root directory (i.e. where the data samples are stored).
* `event_dir`: Path to the event logs.

## `vis_rve.py`

This script loads a dataset of synthetic RVEs and corresponding labels from a directory specified in `parameters.yaml`, then generates PNG visualizations of each RVE with a title indicating the ferrite phase ratio. The images are saved in an `img/` subdirectory inside the data root. The script relies on `visualization_utils.py` from the `../utils `folder for rendering.

In the following, I will explain the key steps and note any important details:

#### Parameter loading

1. **Reads `parameters.yaml`** – Uses `yaml.safe_load()` and verifies that the parsed content is a dictionary; raises a `TypeError` otherwise.

2. **Extracts dataroot** – Expands the user home path and creates the `img/` directory if it does not already exist.

#### Label loading and file iteration

1. **Loads `labels.npy`** – Assumes the file exists in dataroot.

2. **Lists files naturally** – Uses `natsorted` to order all files in dataroot (excluding `labels.npy`) in a human‑friendly order (ok chatGPT??).

3. **Processes each RVE file** – Loads the numpy array, calls `vu.visualize_tensor()` to get a PNG array, then displays and saves the image.

#### Visualization and saving

1. **Renders the RVE** – `vu.visualize_tensor(rve)` returns an image array suitable for `matplotlib`.

2. **Adds a formatted title** – The ferrite phase ratio is computed as `labels[cnt] * 1e-2` and rounded to three decimals.

3. **Saves the PNG** – Each file is written as `rve_<counter>.png` inside the `img/` directory. The counter increments after each successful save.

#### PyVista backend setup

Sets `pv.global_theme.jupyter_backend = 'static'` to force PyVista to generate static PNG images, which is essential for headless environments or when taking screenshots in Jupyter notebooks.

## `vis_loss.py`

This script loads TensorBoard event logs from a directory specified in parameters.yaml, extracts scalar metrics such as generator loss, critic loss, gradient penalty, and real/fake sample losses, and generates a 2×2 summary plot with smoothed curves and interpretation guidelines. The resulting figure is saved as loss_curves.png inside the img/ subdirectory of the data root.

In the following, I will explain what every function does and write down comments worthy of remembering:

#### `find_event_file(event_dir)`

1. **Searches for the TensorBoard event file** – Lists all files in `event_dir` and returns the first one whose name starts with `"events.out.tfevents"`.

2. **Raises clear errors** – `FileNotFoundError` if the directory does not exist or no event file is found.

####` read_tensorboard_logs(log_path, tag)`

1. **Loads a specific scalar tag** – Uses `EventAccumulator` to reload the event file, then extracts step and value lists for the given tag.

2. **Returns two parallel lists** – steps and values, ready for plotting.

#### `print_available_tags(log_path)`

1. **Introspects the event file** – Prints all available tags grouped by category (scalars, histograms, images, etc.).

2. **Helpful for debugging** – Allows the user to verify which metrics were actually logged (e.g., `Loss/Critic`, `GP`, `Loss/D_fake`).


#### `moving_avg(data, window_size=10)`

1. **Computes a centered moving average** – Pads the data with edge values `(mode='edge')` to avoid shortening the series, then applies a box‑filter convolution.

*Important – The returned array has the same length as the input, but the first and last `window_size-1` points are edge‑padded. The script later slices `[9:-9]` to remove the padded edges when plotting the smoothed line.*

#### `main()`

1. **Loads parameters** – Reads `parameters.yaml`, expects keys `event_dir` (path to TensorBoard logs) and dataroot (where `img/` will be created).

2. **Prints interpretation guidelines** – Hard‑coded advice for interpreting generator loss (plateau expected), critic loss (stable negative values), gradient penalty (should stay near zero), and the real‑vs‑fake gap (positive, with fake trending upward).

3. **Locates event file and shows available tags** – Calls `find_event_file` and `print_available_tags`.

4. **Reads five scalar series** – `Loss/Critic`, `Loss/D_fake`, `Loss/D_real`, `Loss/Generator` and `GP`

5. **Creates a 2×2 matplotlib grid** – Subplots:
    0. Generator loss with centered moving average and a zero baseline.
    1. Critic loss with moving average and zero line.
    2. Gradient penalty (no smoothing) with zero line.
    3. Real vs. fake sample losses; shades the area between them (the “distinction gap”) and adds a zero line.

6. **Saves the figure** – Path: `{dataroot}/img/loss_curves.png` (directory created earlier by `visualize_rves.py` or automatically if missing).

7. **Displays the plot** – `plt.show()`.