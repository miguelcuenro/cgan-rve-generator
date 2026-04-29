# cGAN

Here is the required code to succesfully create, train and run the cGAN model. Like all other folders `parameters.yaml` contains all (hyper-)parameters to configure the neural network, which I will explain in a minute. Contrary to what its name indicates, `cgan_creator.py` not only creates the cGAN, but it also contains the necessary code to train it (you can set this up in the parameters file). Then, after training, just run `op_table.py` to generate synthetic RVEs. Other scripts, such as `cGAN.py` and `visualization.py`, are essential for the whole pipeline to work.

## Parameter file

There are many variables and paths stored in `parameters.yaml`, so buckle up while we walk through them carefully!

* `dataroot`: Path to root directory.
* `num_of_workers`: Number of simultaneous loads of data into the RAM.
* `batch_size`: Batch size during training.
* `img_size`: Spatial size of the training volumes (every volume will be resized to this).
* `num_channels`: Number of channels in the training samples.
* `gen_num_feature_maps`: Number of feature maps in the generator.
* `gen_dropout_rate`: Fraction of neurons turned off every step (to avoid overfitting).
* `dis_num_feature_maps`: Number of feature maps in the discriminator
* `dis_dropout_rate`: Fraction of neurons turned off every step (to avoid overfitting).
* `num_epochs`: Number of training epochs.
* `learning_rate_disc`: Adjusts the learning rate of the discriminator.
* `learning_rate_gen`: Adjusts the learning rate of the generator.
* `d_loop`: Factor, which decides how many times the critic is trained for each gen training step.
* `beta1`: Exponential decay rate for the first moment (momentum) in Adam optimizer.
* `beta2`: Exponential decay rate for the second moment (RMSprop‑like adaptive scaling) in Adam optimizer.
* `ngpu`: Number of available GPUs (0 for cpu mode).
* `lambda_penal`: Lambda multiplier for the gradient penalty.
* `sigma`: Scaling factor of the statistical matching loss (not used in my code).
* `from_checkpoint`: Whether the cGAN should be trained from a given checkpoint (1) or not (0).
* `save_checkpoints`: Wheter the results from training should be saved or not (common sense says yes!)
* `enable_sampling`: Generate and visualize samples on fixed noise during training (to visualize the development).
* `checkpoint`: Path to the training checkpoint (to continue training or sample).
* `number_of_samples`: Number of generated RVEs.
* `root`: Directory for the generated samples.
* `sampling_dir`: NOT USED AT ALL --> REMOVE IT!

## `cgan_creator.py`

As said before this script runs the required code to create and/or train the model (selected through `from_checkpoints`). During training you can decide wether you want to save checkpoints (setting `save_checkpoints` to 1) and if you want to generate samples during training (setting `generate samples` to 1) on fixed noise to visualize the development of the model. Note that turning `save_checkpoints` off does not save any models, virtually deleting the model the moment it stops training!

In the following, I will explain what every function does and write down comments worthy of remembering:

#### `class CustomDataset(Dataset)`

A minimal PyTorch Dataset wrapper to pair data samples with their labels.

#### `find_npy_files`

1. **Walks the directory tree** – Recursively traverses `root_dir` using `os.walk()`.

2. **Filters for phase_grid files** – Checks each file name. Currently matches only exact name `phase_grid.npy` (not augmented files like `phase_grid_rotated_0.npy`).

3. **Builds full file paths** – For each matching file, joins the root path with the file name.

4. **Appends to list** – Adds the full path to the `npy_files` list.

5. **Returns the list** – Provides all discovered `phase_grid.npy` paths for further processing.

#### `load_data`

1. **Iterates through `.npy` files** – Loops over a list of phase grid file paths.

2. **Validates array size** – Checks if the loaded array size equals `img_size^3` (e.g., 32^3, 64^3). If not, skips the file.

3. **Reshapes phase grid** – Converts valid arrays to shape `(1, num_channels, img_size, img_size, img_size)` and appends to data list.

4. **Matches corresponding label** – Looks for `label.npy` in the same directory. If found, loads it and appends to label list. If missing, removes the just‑added grid and prints a warning.

5. **Checks label shape consistency** – Tracks unique label shapes. Raises an error if multiple different shapes are found, printing the shapes for debugging.

6. **Concatenates all data** – Merges all valid phase grids into a single numpy array.

7. **Repeats labels** – Repeats each label to match the number of data samples (useful when multiple rotated grids share the same label file).

8. **Normalizes labels** – Applies min‑max scaling to each label column to the range [0, 1]. Avoids division by zero by setting the range to 1.0 when min equals max.

9. **Converts to PyTorch tensors** – Converts data and labels to torch.DoubleTensor.

10. **Returns a CustomDataset** – Wraps the tensors in a CustomDataset for use with a PyTorch DataLoader.