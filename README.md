# Generating RVE's with a cGAN

This repository contains code and resources for generating three-dimensional Representative Volume Elements (RVEs) with predefined material characteristics using a conditional Generative Adversarial Network (cGAN)... or at least it should :

## Description



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

On the other side, accessing all the functionalities of the cGAN requires an explanation a little more extensive. Besides the countless hyperparameters there are a few interesting parameters I will explain in the following:

| Parameter  | Description |
| ------------- | ------------- |
| `dataroot`  | Directory where the processed data is contained.  |
| `from_checkpoint`  | Takes `0` if you do not wish to train from a given checkpoint and `1` otherwise.  |
| `save_checkpoints` | Likewise: `0` if no checkpoints should be saved and `1` otherwise (by default it is set to save the checkpoints, as this is the goal of training the cGAN). |
| `enable_sampling` | `0` if no samples should be taken during training, `1` otherwise. |
| `checkpoint` | Path to the checkpoint you want to train from. |

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

If there are any issues, please do not doubt on contact me. :)

## Authors

Miguel Cuenca Rodríguez

based on the bachelor thesis of Xavier Mertz.