# Generating RVE's with a cGAN

This repository contains code and resources for generating three-dimensional Representative Volume Elements (RVEs) with predefined material characteristics using a conditional Generative Adversarial Network (cGAN)... or at least it should :)

## Getting Started

### Dependencies

* There are no prerequisites for the installation of this program, as a matter of fact, I doubt any installation is required at all!
* However, the python code requires some libraries (such as tensorboard, pyvista, etc.) to run, so we recommend to install those, e.g. with conda, as described in `environment.yml`.

### Executing program

The code is set in such way that it is not necessary for you, the user, to do any modifications to the python scripts in order for it to run. You can access all (hyper)parameters through the corresponding `parameters.yml` and do the required modifications so that it works in your own setup.

In case there is any doubt, to run the code you just need to activate your environment (e.g. `conda activate my-env`) and run the code with python:
```
python mycode.py
```

#### 1. Preprocess the data.
The first step is to preprocess the data with `src/data_processing/phaseExtractor.py`. Within `src/data_processing/parameters.yml` you can state in which directory your data samples are and where do you wish for the processed data to be stored. By default `parameters.yml` looks for data stored in `data/raw_data` and stores the processed output in `data/processed_data`.

#### 2. Train the conditional generative adversarial neural network (cGAN for friends)

TL;DR, To train the neural network the only thing you have to do is run `src/models/cgan_creator.py`.

On the other side, accessing all the functionalities of the cGAN requires an explanation a little more extensive. Besides the countless hyperparameters there are a few interesting parameters I will explain in the following table:

| Parameter  | Description |
| ------------- | ------------- |
| `dataroot`  | Directory where the processed data is contained.  |
| `from_checkpoint`  | Takes `0` if you do not wish to train from a given checkpoint and `1` otherwise.  |
| `save_checkpoints` | Likewise: `0` if no checkpoints should be saved and `1` otherwise (by default it is set to save the checkpoints, as this is the goal of training the cGAN). |
| `enable_sampling` | `0` if no samples should be taken during training, `1` otherwise. |
| `checkpoint` | Path to the checkpoint you want to train from. |

#### 3. Generating RVEs

Once the training of the cGAN (notice this denomination implies a hard earned friendship) is over, one can generate RVEs with `src/models/op_table.py`. The script is runned as usual, which makes use of the same parameter file as before (`src/models/parameters.yml`). I now, will introduce some of the required parameters:

| Parameter  | Description |
| ------------- | ------------- |
| `checkpoint`  | Path to the state of the cGAN you want to use for generation of data.  |
| `root`  | Directory where the generated RVEs are to be stored.  |
| `number_of_samples`  | Self explanatory. |

#### 4. Visualization of the RVEs

If, after the generation, you desire to visualize the data this is also possible! Run `visualization.py` to read the information of the generated RVEs and generate a 3D graphic of it. This script uses the same `src/models/parameters` file (again) and reads the data stored under the directory given by the parameter `sampling_dir`.

## Disclaimer

I am well aware, this code is far from perfect, but I am still working on it.

Further changes will include:
* Ensuring the cGAN works properly (I did not have the chance to try it yet as I am having trouble with the HPC)
* Better control of the RVE's phase ratio
* Better clarity with the `parameters.yml` files
* Improved management of the directories (specially the outputed ones)
* Hyperparameter tuning (e.g. with Optuna)

## Help

If there are any issues, please do not doubt on contacting me. :)

## Authors

Miguel Cuenca Rodríguez

based on the bachelor thesis of Xavier Mertz.