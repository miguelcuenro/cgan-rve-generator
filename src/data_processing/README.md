# Data Pprocessing

`phaseExtractor.py` is a code designed to parse (I think that's how it is called) through  folders and look for `materials.yaml` and `Specs.txt` files and preprocess them to generate samples so the cGAN can safely digest it. The data samples are then augmented so the cGAN has more than enough data samples.

## Parameter file

There are only two variables in the parameter file (`parameters.yml`) that set up the way `phaseExtractor.py` operates:

* `source`: Where your data folders are stored.
* `destination`: Where your new, processed, data samples should be stored.

## Functions

This section is mostly for **ME**, because I am very forgetful. I will explain what every (important) function in the code does so I can remember it quickly, specially when it comes to "kill them Bugs"! (KUDOS to deepseek??)


#### `process_simulation_dir`

1. **Validates required files** – Skips the directory if `material.yaml` or `Specs.txt` is missing.

2. **Loads material phases** – Reads `material.yaml`, maps Ferrite --> 0, other phases --> 1.

3. **Checks grid size** – Loads `grid.vti`; skips if the material array is not exactly 32×32×32.

4. **Builds 3D phase array** – Converts per‑point material indices into a binary phase grid (0/1) and reshapes to (32,32,32).

5. **Creates output subdirectory** – Uses the simulation folder’s base name as subfolder inside `output_dir`.

6. **Saves phase grid** – Writes the (32,32,32) array as `phase_grid.npy`.

7. **Parses Specs.txt** – Extracts 7 values: ferrite %, martensite %, number of bands, and four adjustment factors (size/aspect ratio for both phases).

8. **Saves label vector** – Stores the 7 values as `label.npy`.

9. **Updates global counters** – Increments counter on success; updates band‑zero/band‑positive counters; increments skip counters on failures.

10. **Returns** – The updated total count of successful iterations.

#### `perform_augmentation`


1. **Walks through each subdirectory** – Inside `data_dir`.

2. **Checks for required files** – Looks for `phase_grid.npy` and `label.npy`; skips if either is missing.

3. **Loads the phase array and label array** – From those files.

4. **Extracts the number of bands** – From the 3rd element (index 2) of the label array.

5. **Conditionally applies rotations** – If `num_bands >= 0` (always true for non‑negative counts; effectively all cases), it rotates the 3D phase array.

6. **Rotates along three axes** – For `axis = 0,1,2`, performs a single 90° rotation (`k=1`) using axes `(axis, (axis+1)%3)`, i.e., three orthogonal rotation planes.

7. **Saves each rotated array** – As `phase_grid_rotated_0.npy`, `phase_grid_rotated_1.npy`, `phase_grid_rotated_2.npy` inside the same subdirectory.

8. **Prints a confirmation** – Reports that augmentation was done for that subdirectory.

#### `traverse_directories`


1. **Initializes a counter** – To track successful processings.

2. **Lists all entries** – In `input_dir` and sorts them alphabetically.

3. **Iterates over each entry** – Builds the full path for each subdirectory.

4. **Calls `process_simulation_dir`** – On that subdirectory, passing the `output_dir` and the current counter.

5. **Updates the counter** – With the value returned by `process_simulation_dir` (which increments on successful processing).

6. **Returns** – The final counter, i.e., total number of successfully processed directories.