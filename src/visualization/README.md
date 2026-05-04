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