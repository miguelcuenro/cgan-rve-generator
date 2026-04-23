# Data Pprocessing

`phaseExtractor.py` is a code designed to parse (I think that's how it is called) through  folders and look for `materials.yaml` and `Specs.txt` files and preprocess them to generate samples so the cGAN can safely digest it. The data samples are then augmented so the cGAN has more than enough data samples.

## Parameter file

There are only two variables in the parameter file (`parameters.yml`) that set up the way `phaseExtractor.py` operates:

* `source`: Where your data folders are stored.
* `destination`: Where your new, processed, data samples should be stored.

## Functions

This section is mostly for **ME**, because I am very forgetful. I will explain what every (important) function in the code does so I can remember it quickly, specially when it comes to "kill them Bugs"!

#### `process_simulation_dir`

1. **Validates required files** – Skips the directory if material.yaml or Specs.txt is missing.

2. **Loads material phases** – Reads material.yaml, maps Ferrite --> 0, other phases --> 1.

3. **Checks grid size** – Loads grid.vti; skips if the material array is not exactly 32×32×32.

4. **Builds 3D phase array** – Converts per‑point material indices into a binary phase grid (0/1) and reshapes to (32,32,32).

5. **Creates output subdirectory** – Uses the simulation folder’s base name as subfolder inside output_dir.

6. **Saves phase grid** – Writes the (32,32,32) array as phase_grid.npy.

7. **Parses Specs.txt** – Extracts 7 values: ferrite %, martensite %, number of bands, and four adjustment factors (size/aspect ratio for both phases).

8. **Saves label vector** – Stores the 7 values as label.npy.

9. **Updates global counters** – Increments counter on success; updates band‑zero/band‑positive counters; increments skip counters on failures.

10. **Returns** – The updated total count of successful iterations.