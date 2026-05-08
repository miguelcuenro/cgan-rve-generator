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

#### `train()`

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

#### `load_checkpoint()`

1. **Loads checkpoint from disk** – Uses `torch.load()` with `map_location='cpu'` to load the saved `.pth` file.

2. **Prints success message** – Confirms the model was loaded successfully.

3. **Extracts generator state dict** – Retrieves `generator_state_dict` from the checkpoint.

4. **Updates generator weights** – Iterates through loaded state dict, copies parameters that exist in the current model (skips any missing keys with a warning).

5. **Loads updated dict into generator** – Calls `load_state_dict()` with the merged state dict.

6. **Extracts critic (discriminator) state dict** – Retrieves `discriminator_state_dict` from the checkpoint.

7. **Updates critic weights** – Same key‑by‑key copying, skipping missing keys and printing warnings.

8. **Loads updated dict into critic** – Calls `load_state_dict()` with the merged critic state dict.

9. **Restores optimizers** – Loads the saved state dicts for both generator and critic optimizers (`optimizerG_state_dict`, `optimizerD_state_dict`).

10. **Retrieves last completed epoch** – Uses `checkpoint.get('epoch', 0)` to obtain the epoch number to resume from, defaulting to 0 if missing.

11. **Prints resuming epoch** – Informs the user from which epoch training will continue.

#### `train_from()`

1. **Accepts checkpoint path and training flags** – Takes `checkpoint_path`, `save_checkpoints`, and `enable_sampling` as arguments.

2. **Loads the checkpoint** – Calls `self.load_checkpoint(checkpoint_path)` to restore model weights, optimizer states, and the last completed epoch.

3. **Resumes training** – Calls `self.train(save_checkpoints, enable_sampling)` to continue training from the restored state, respecting the user's flags for checkpointing and sampling.

### `class Critic(nn.Module)`

#### `__init__()`

1. **Stores critic configuration** – Saves `num_channels`, `num_feature_maps`, `img_size`, and `dis_dropout_rate` as instance attributes.

2. **Computes number of layers** – Calculates `num_layers = log2(img_size)` and generates a list of feature multipliers `features = [2^0, 2^1, ..., 2^(num_layers-3)]`.

3. **Defines a pre‑layer (conv3d)** – A `nn.Conv3d` layer that takes `num_channels+1` input channels (sample + label projection) and outputs `num_channels` channels, using kernel 3, stride 1, padding 1, no bias.

4. **Creates initial convolutional layer** – `nn.Conv3d` from `num_channels` to `num_feature_maps`, with kernel 4, stride 2, padding 1, no bias.

5. **Builds hidden layers dynamically** – Iterates `(num_layers - 3)` times, each time increasing channels by factor 2 (using `features` list). Each hidden layer is a `Sequential` block containing:

    * `Conv3d` (kernel 4, stride 2, padding 1, no bias)

    * `BatchNorm3d`

    * `LeakyReLU(0.2)`

    * `Dropout3d` with `dis_dropout_rate`

6. **Stores hidden layers as `ModuleList`** – Ensures PyTorch properly registers the modules.

7. **Defines final convolutional layer** – `Conv3d` that reduces spatial dimensions to 1×1×1 and outputs a single scalar (critic score), using kernel 4, stride 2, padding 0, no bias.

8. **Creates label embedding network** – A small MLP: `Linear(1 --> 512) --> LeakyReLU --> Linear(512 --> 32768) --> LeakyReLU`, all as `double()` tensors. The output size `32768 = 32^3` is designed to be reshaped into a 3D cube for concatenation with the sample.

#### `forward(self, sample, label)`

1. **Projects the label** – Passes the input label through `self.label_layers` (MLP) to produce a feature vector.

2. **Reshapes label features into a 3D cube** – Uses `.view(-1, 1, 32, 32, 32)` to create a tensor with 1 channel and spatial dimensions matching the input sample (assumed 32×32×32).

3. **Concatenates sample and label cube** – Stacks the sample and `lp_3d` along the channel dimension `(dim=1)`, increasing channels to `num_channels + 1`.

4. **Passes through pre‑layer** – Applies the 3D convolution self.prelayer, followed by `LeakyReLU(0.2)` activation.

5. **Passes through initial layer** – Applies `self.initial_layer` (Conv3d), followed by `LeakyReLU(0.2)`.

6. **Processes through hidden layers** – Sequentially passes the tensor through each layer in `self.hidden_layers` (each containing Conv3d + BatchNorm + LeakyReLU + Dropout3d).

7. **Final layer output** – Applies `self.final_layer` (Conv3d) to produce a single scalar score (critic output) for each sample in the batch.

8. **Returns the critic score** – Output shape is `(batch_size, 1, 1, 1, 1)`.

### `Generator(nn.Module)`

#### `__init__()`

1. **Stores generator configuration** – Saves `num_channels`, `num_feature_maps`, `img_size`, and `gen_dropout_rate` as instance attributes.

2. **Computes layer count and feature multipliers** – Calculates `num_layers = log2(img_size)`, creates list `[2^0, 2^1, …, 2^(num_layers-3)]`, then reverses it for progressive upsampling.

3. **Adds reflection padding** – `nn.ReflectionPad3d(1)` to maintain spatial dimensions during convolutions.

4. **Defines initial transposed convolution layer** – `nn.ConvTranspose3d` with `in_channels=2` (noise + label projection), `out_channels = num_feature_maps` // features[0], kernel 4, stride 2, padding 4, bias False. Followed by BatchNorm3d.

5. **Builds hidden transposed convolution layers** – Loops `num_layers - 3` times, each layer reduces channels (using integer division `num_feature_maps // features[i]`). Each hidden layer consists of:
    * `ConvTranspose3d` (kernel 4, stride 2, padding 2, no bias)

    * `BatchNorm3d`

    * `LeakyReLU(0.2)`
    
    * `Dropout3d` with the provided `gen_dropout_rate`

6. **Stores hidden layers as ModuleList** – Ensures PyTorch registers all modules correctly.

7. **Defines final transposed convolution layer** – `ConvTranspose3d` with `in_channels = i_out` (last hidden layer's output channels), `out_channels = num_channels`, kernel 3, stride 1, padding 2, no bias.

8. **Creates label embedding network** – A small MLP: `Linear(1 --> 16) --> LeakyReLU --> Linear(16 → 64) --> LeakyReLU`, all as `double()` tensors. The output size 64 is later reshaped into a `4^3` cube for concatenation with noise.

#### `forward()`

1. **Projects the label** – Passes label through `self.label_layers` (MLP) to produce a 64‑dimensional feature vector.

2. **Reshapes label features into a 3D cube** – Uses `.view(-1, 1, 4, 4, 4)` to create a tensor with 1 channel and spatial dimensions `4×4×4`.

3. **Concatenates noise and label cube** – Stacks the input `noise` (latent vector, shape: batch × 1 × 4×4×4) and `lp_3d` along the channel dimension (dim=1), resulting in `in_channels = 2`.

4. **Applies initial padding** – Uses `self.padding` (ReflectionPad3d(1)) to expand spatial dimensions before the first transposed convolution.

5. **Passes through initial layer** – Applies `self.initial_layer` (ConvTranspose3d --> BatchNorm), then a `LeakyReLU(0)` activation.

6. **Applies padding again** – Adds reflection padding before the hidden layers.

7. **Processes through hidden layers** – For each layer in `self.hidden_layers`, applies the layer (ConvTranspose3d → BatchNorm → LeakyReLU(0.2) → Dropout), then adds reflection padding. This occurs twice for the default architecture (num_layers = 5, so 2 hidden layers).

8. **Final transposed convolution** – Applies `self.final_layer` (ConvTranspose3d with kernel 3, stride 1, padding 2) followed by `LeakyReLU(0)`.

9. **Final padding and sigmoid** – Applies reflection padding one more time to reach the target spatial size (32×32×32), then passes through `torch.sigmoid()` to clamp output values between 0 and 1.

10. **Returns generated image** – Output tensor of shape `(batch_size, num_channels, 32, 32, 32)`.
## `op_table.py`

This script loads our (pre-trained) cGAN and generates an arbitrary, chosen by you, number of synthetic RVEs. The generated labels and samples then get stored as numpy arrays.

#### `class CustomDataset(Dataset)`

1. **Inherits from `torch.utils.data.Dataset`** – A standard PyTorch dataset wrapper.

2. **Initializes with data and labels** – Stores data and labels tensors as instance attributes.

3. **Implements `__len__` method** – Returns the number of samples in the dataset.

4. **Implements `__getitem__` method** – Retrieves the sample and label at a given index `idx` as a tuple `(data[idx], labels[idx])`.

#### `find_npy_files()`

1. **Walks the directory tree** – Recursively traverses `root_dir` using `os.walk()`.

2. **Filters for `phase_grid.npy` files** – Checks each file name exactly matches `'phase_grid.npy'` (currently does not match augmented files like `phase_grid_rotated_0.npy`).

3. **Builds full file paths** – For each matching file, joins the root path with the file name.

4. **Appends to a list** – Adds the full path to the `npy_files` list.

5. **Returns the list** – Provides all discovered `phase_grid.npy` paths for further processing.

#### `load_data()`

Note that the data we upload doesn't get used AT ALL! Except it does, the cGAN model requires data to get instantiated (Will I change that in the future? Maybe, but not now that's for sure).

1. **Initializes empty lists and a set** – `data_list` for 3D volumes, `label_list` for corresponding labels, and `label_shapes` to track unique label shapes.

2. **Iterates over each .npy file path** – Loads the array using `np.load(allow_pickle=True)`.

3. **Checks volume size** – If the data size matches `img_size**3`, reshapes it to `(1, num_channels, img_size, img_size, img_size)` and appends to `data_list`.

4. **Looks for associated label file** – Constructs `label.npy` path in the same directory as the volume.

5. **Handles missing labels** – If label file exists, loads it, records its shape in `label_shapes`, and appends to `label_list`. If missing, removes the previously added volume from `data_list` (via `pop()`) and skips the file.

6. **Validates data existence** – After the loop, raises an error if no valid data was found.

7. Checks label shape consistency – If multiple distinct label shapes are found, prints them and raises an error.

8. **Concatenates all data** – Uses `np.concatenate()` to stack all volumes into a single array data_np.

9. **Repeats labels to match data count** – `np.repeat()` ensures each label is duplicated across all corresponding volumes (handles case where multiple volumes share the same label file).

10. **Records global label min/max** – Stores `labels_min` and `labels_max` (as global variables) for later de‑normalization when generating new samples.

11. **Converts to PyTorch tensors** – Creates `data_tensor` and `labels_tensor` as `double()` tensors.

12. **Returns a CustomDataset instance** – Wraps the tensors in the previously defined CustomDataset class.

#### `log_memory_usage()`

It doesn't get called anywhere tho, but that doesn't mean it does not deserve its very own description :).

1. **Takes a step argument** – Used to identify when the memory log is recorded.

2. **Queries CUDA memory stats** – Calls `torch.cuda.memory_allocated()` to get the current allocated memory (bytes).

3. **Gets reserved memory** – Calls `torch.cuda.memory_reserved()` to get the total memory reserved by the caching allocator.

4. **Prints the information** – Outputs a formatted string with the step number, allocated memory, and reserved memory.

5. **Useful for debugging** – Helps track GPU memory usage during training or inference.

#### `main()`

1. **Clears GPU cache** – Calls `torch.cuda.empty_cache()` at the start to free unused memory.

2. **Loads hyperparameters** – Reads `parameters.yaml` and assigns values to variables (dataroot, batch_size, img_size, num_channels, model architectures, training settings, etc.). Makes `img_size` and `num_channels` global.

3. **Prepares dataset** – Calls `find_npy_files(dataroot)` and `load_data(npy_files)` to create a CustomDataset with 3D volumes and their labels.

4. **Instantiates the cGAN model** – Creates a `cGAN.DCWCGANGP` object using all loaded parameters and the dataset.

5. **Prints dataset length and device info** – Outputs the number of samples and the device (CPU/CUDA) the model is using.

6. **Clears cache again** – Calls `torch.cuda.empty_cache()` before loading the checkpoint.

7. **Loads pre‑trained checkpoint** – Uses `operated_cgan.load_checkpoint(checkpoint_path)` to restore model weights and optimizer states.

8. **Clears cache once more** – Final cache clearing before generation.

9. **Creates a timestamped output directory** – Constructs a path using `parameters['root']` and current date/time, then creates the directory.

10. **Generates synthetic samples** – Loops `number_of_samples` times:
    * Draws random latent noise `binary_noise` (shape based on `num_of_z`).
    
    * Draws random label `label_values` (uniform in [0,1]).

    * Passes noise and label through `operated_cgan`.gen to produce a raw image.

    * Rounds the output to binary values with `torch.round()`.

    * Saves the volume as a `.npy` file (`number_i.npy`).

    * Stores the raw label value in `label_list`.

11. **De‑normalizes labels** – Converts `label_list` to a numpy array and rescales from [0,1] to the original label range using stored `labels_min` and `labels_max`.

12. **Saves the labels** – Writes the de‑normalized labels to `labels.npy` in the same directory.

13. **Prints completion message** – Confirms that all samples have been generated.