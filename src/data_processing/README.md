# Data Pprocessing

`phaseExtractor.py` is a code designed to parse (I think that's how it is called) through  folders and look for `materials.yaml` and `Specs.txt` files and preprocess them to generate samples so the cGAN can safely digest it. The data samples are then augmented so the cGAN has more than enough data samples.

## Parameter file

There are only two variables in the parameter file (`parameters.yml`) that set up the way `phaseExtractor.py` operates:

* `source`: Where your data folders are stored.
* `destination`: Where your new, processed, data samples should be stored.

## Functions

This section is mostly for **ME**, because I am very forgetful. I will explain what every (important) function in the code does so I can remember it quickly, specially when it comes to "kill them Bugs"!

#### `process_simulation_dir`