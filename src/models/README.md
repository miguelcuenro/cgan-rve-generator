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

#### `find_npy_files()`

1. **Walks the directory tree** – Recursively traverses `root_dir` using `os.walk()`.

2. **Filters for phase_grid files** – Checks each file name. Currently matches only exact name `phase_grid.npy` (not augmented files like `phase_grid_rotated_0.npy`).

3. **Builds full file paths** – For each matching file, joins the root path with the file name.

4. **Appends to list** – Adds the full path to the `npy_files` list.

5. **Returns the list** – Provides all discovered `phase_grid.npy` paths for further processing.

#### `load_data()`

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

#### `main()`

1. **Clears GPU cache** – Calls `torch.cuda.empty_cache()` to free GPU memory, useful for cluster environments to avoid out-of-memory issues.

2. **Starts virtual frame buffer** – `pv.start_xvfb()` initializes a virtual display for rendering screenshots on headless servers.

3. **Loads hyperparameters** – Reads `parameters.yaml` and assigns values to variables like `batch_size`, `num_epochs`, `learning rates`, etc.

4. **Prepares the dataset** – Uses `find_npy_files(dataroot)` to locate all `phase_grid.npy` files, then `load_data()` to create the dataset with specified `img_size` and `num_channels`.

5. **Instantiates the GAN model** – Creates the `DCWCGANGP` object (conditional Wasserstein GAN with gradient penalty) using all loaded parameters.

6. **Starts or resumes training** – If `from_checkpoint` is `False`, calls `cgan.train()`; otherwise calls `cgan.train_from()` with the checkpoint path. Both methods respect `save_checkpoints` and `enable_sampling` flags.

## `CGAN.py`

This script contains the conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP) we use to generate the RVEs. The, here-defined, classes get called by `cgan_creator.py`, so there is little to worry about this code.

### `class DCWCGANGP`

#### `__init__()`

1. **Stores training hyperparameters** – batch size, epochs, learning rates, optimiser betas, gradient penalty weight, etc.

2. **Sets up device** – uses CUDA if available and `ngpu ≥ 1`, otherwise CPU.

3. **Instantiates Generator and Critic** – creates the two networks with the given feature maps and dropout rates, moves them to the device as `double()`, and applies `init_weights`.

4. **Creates DataLoader** – wraps the dataset with the specified batch size and shuffling.

5. **Initialises fixed test inputs** – `fixed_label` (random) and `fixed_z` (rounded random noise) for consistent sampling during training.

6. **Prepares storage** – empty lists for loss tracking and a step counter.

7. **Configures Adam optimizers** – one for the critic, one for the generator.

8. **Saves hyperparameters as a string** – for logging to TensorBoard.

#### `init_weights()`

1. **Identifies layer type** – Checks the class name of the module.

2. **Initializes convolutional layers** – If the class name contains `'Conv'`, applies normal initialization with mean `0.0` and standard deviation `0.02` to the weight data.

3. **Initializes batch norm layers** – If the class name contains `'BatchNorm'`, initializes weights with normal distribution (mean `1.0`, std `0.02`) and sets bias to constant `0`.

4. **Ignores other layers** – Leaves all other module types unchanged.

#### `compute_gradients()`

1. **Generates random interpolation weights** – Creates an alpha tensor of random values (same shape as real samples) on the device, used to mix real and fake samples.

2. **Computes interpolated samples** – Linearly interpolates between real and fake samples: `interpolates = alpha * real + (1-alpha) * fake`. Sets `requires_grad=True` to track gradients.

**3. Passes interpolates through critic** – Computes critic scores for the interpolated samples.

4. **Prepares gradient output tensor** – Creates a fake tensor of ones with the same batch size as `real_samples` (shape: `[batch, 1,1,1,1]`) to serve as `grad_outputs`.

5. **Calculates gradients** – Uses `torch.autograd.grad()` to compute gradients of critic outputs with respect to the interpolated inputs.

6. **Computes gradient penalty** – Takes the norm of the gradients (L2 norm), subtracts 1, squares the result, and averages across the batch.

7. **Returns penalty term** – Returns the computed gradient penalty (scalar tensor).

#### `train()``

1. **Prints start message and creates timestamp** – Records training start time and generates a unique timestamp for the log directory.

2. **Creates log directories and TensorBoard writer** – If either `save_checkpoints` or `enable_sampling` is `True`, creates a `training_logs/timestamp` folder and initializes a `SummaryWriter`.

3. **Sets up checkpoint and sample subdirectories** – If `save_checkpoints` is `True`, creates a `checkpoint_dir` and logs hyperparameters to TensorBoard. If `enable_sampling` is True, creates a `sample_dir`.

4. **Initializes sampling frequency and counter** – `sampling_freq = 150` (changes to 500 after epoch 51), `sampling_counter = -1`.

5. **Determines starting epoch** – Uses `self.start_epoch` if resuming from a checkpoint (default 0).

6. **Enters epoch loop** – Iterates from `start_epoch` to `num_epochs`.

7. **Iterates over batches** – For each batch:

    * Unpacks data and labels, moves to device.

    * Generates random noise and creates fake images via generator.

    * At epoch 51, increases `sampling_freq` to 500.

    * Conditionally saves samples (both random and fixed) every `sampling_freq` batches, logs them to TensorBoard.
    
    * After 5000 total steps, reduces `d_loop` to 5.

8. **Trains the critic (discriminator) for `d_loop` iterations:**

    * Computes real and fake critic outputs.

    * Calculates loss d_loss = -(mean(real) - mean(fake)).

    * Computes gradient penalty and adds it scaled by lambda_penal.

    * Logs losses and penalty to TensorBoard if save_checkpoints.

    * Backpropagates total critic loss and updates optimizer.

9. **Trains the generator once:**

    * Computes critic outputs on fake images.

    * Generator loss g_loss = -mean(outputs).

    * Logs generator loss to TensorBoard if save_checkpoints.

    * Backpropagates and updates generator optimizer.

10. **Saves checkpoint at epoch end** – Condition: last epoch, every hundred epochs `(epoch % 1 == 100)`, or every 1000 steps. Saves model states, optimizer states, and losses.

11. **After all epochs** – If `save_checkpoints` is `True`, logs training end time and duration to TensorBoard.

12. **Closes TensorBoard writer** – If either `save_checkpoints` or `enable_sampling` was `True`.

13. **Prints completion message.**