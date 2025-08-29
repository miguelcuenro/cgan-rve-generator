import pyvista as pv
import yaml
import os
import numpy as np

# Load the parameters file
with open('parameters.yaml', 'r') as dictionary:
    parameters = yaml.safe_load(dictionary)

# Raise an error if parameters.yaml is NOT a dictionary
if not isinstance(parameters, dict):
    raise TypeError("parameters.yaml did not parse to a dictionary. Please check the file's structure.")

# Check for required keys
required_keys = ['source', 'destination']
missing_keys = [key for key in required_keys if key not in parameters]

if missing_keys:
    raise KeyError(f"Missing required keys in config.yaml: {missing_keys}")

# Path to the data folders
raw_data_dir = parameters['source']
processed_data_dir = parameters['destination']

# Raise an error if raw_data_dir was NOT found
if not os.path.exists(raw_data_dir):
    raise FileNotFoundError(f"Source data directory not found: {raw_data_dir}")

# Raise an error if processed_data_dir was NOT found
if not os.path.exists(processed_data_dir):
    raise FileNotFoundError(f"Destination data directory not found: {processed_data_dir}")

# Create parent directory if it does not exist
os.makedirs(processed_data_dir, exist_ok=True)

# Initialize counts
number_of_skipped = 0
number_of_NOT_skipped = 0
number_of_skipped_size = 0
number_of_skipped_files = 0
number_of_bands_zero = 0
number_of_bands_more_than_zero = 0

def process_simulation_dir(simulations_dir: str, output_dir: str, counter: int) -> int:
    '''
    Processes the simulation directory:
    - Checks for required files
    - Extracts and converts phase information
    - Loads simulation specs. and creates a label vector
    - Saves results to structured subdirectories

    Parameters
    ----------
    simulations_dir : str
        The path to the directory holding the sample data.

    output_dir: str
        The path to the directory where the processed data should be stored.

    counter: int
        Number of iterations

    Returns
    ----------
    counter: int
        Total number of successful iterations (till now).
    '''
    global number_of_skipped, number_of_NOT_skipped, number_of_skipped_size 
    global number_of_skipped_files, number_of_bands_zero
    global number_of_bands_more_than_zero

    files = os.listdir(simulations_dir)

    # Early break if the requirements are not met
    if 'material.yaml' not in files or 'Specs.txt' not in files:
        print(f'Skipping {simulations_dir} as material.yaml or Specs.txt not found.')
        number_of_skipped_files += 1
        return counter

    material_file_path = os.path.join(simulations_dir, 'material.yaml')
    specs_file_path = os.path.join(simulations_dir, 'Specs.txt')

    
    with open(material_file_path, 'r') as ym:
        ym = yaml.safe_load(ym)

def traverse_directories(input_dir: str, output_dir: str) -> int:
    '''
    Walk through each simulation directory in the provided base path, calling process_simulation_dir on each.

    Parameters
    ----------
    input_dir: str
        The path to the directory holding the data to be processed.

    output_dir: str
        The path to the directory where the processed data should be stored.

    Returns
    ----------
    counter: int
        Number of iterations.
    '''
    counter = 1
    for dir_name in sorted(os.listdir(input_dir)):
        full_dir_path = os.path.join(input_dir, dir_name)
        counter = process_simulation_dir(full_dir_path, output_dir, counter)
    return counter
