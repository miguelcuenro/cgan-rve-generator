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

    mat = ym['material']
    phase_list = []

    for m in mat:
        phase = m['constituents'][0]['phase']
        if phase.startswith('Ferrite'):
            phase_list.append(0)
        else:
            phase_list.append(1)

    grid = pv.read(simulations_dir + '/grid.vti')
    phase_array = np.zeros(grid['material'].__len__())

    # Check  if the phase array can be reshaped to 32x32x32
    if phase_array.size != 32 * 32 * 32:
        print(f"Skipping {simulations_dir} as the grid size is not 64x64x64.")
        number_of_skipped_size += 1
        return counter

    for k in range(len(phase_list)):
        if phase_list[k] == 0:
            points = grid['material'].flatten(order='F') == k
            phase_array[points] = 0
        else:
            points = grid['material'].flatten(order='F') == k
            phase_array[points] = 1

    phase_array = phase_array.reshape((32, 32, 32))
    number_of_NOT_skipped += 1

    # Extract simulation number from the raw_data_dir
    number = os.path.basename(simulations_dir)
    sub_dir_name = str(number)

    # Create the new directory for storing phase information
    new_dir_path = os.path.join(processed_data_dir, sub_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    # Save the phase array to a numpy file
    phase_file_path = os.path.join(new_dir_path, 'phase_grid.npy')
    np.save(phase_file_path, phase_array)

    try:
        # Read and extract information from Specs.txt
        with open(specs_file_path, 'r') as specs_file:
            specs_content = specs_file.readlines()

        # Initialize variables
        ferrite_percentage = None
        martensite_percentage = None
        num_bands = 0
        adjustment_martensite_size = None
        adjustment_martensite_aspect_ratio = None
        adjustment_ferrite_size = None
        adjustment_ferrite_aspect_ratio = None

        # Parse the Specs.txt file
        for line in specs_content:
            if 'Percentage of Ferrite' in line:
                ferrite_percentage = float(line.split(': ')[1].strip('%\n'))
            elif 'Percentage of Martensite' in line:
                martensite_percentage = float(line.split(': ')[1].strip('%\n'))
            elif 'Number of Bands' in line:
                num_bands = int(float(line.split(': ')[1].strip()))
            elif 'Adjustment of martensite size' in line:
                adjustment_martensite_size = float(line.split('x ')[1].strip())
            elif 'Adjustment of martensite aspect ratios' in line:
                adjustment_martensite_aspect_ratio = float(line.split('x ')[1].strip())
            elif 'Adjustment of ferrite size' in line:
                adjustment_ferrite_size = float(line.split('x ')[1].strip())
            elif 'Adjustment of ferrite aspect ratios' in line:
                adjustment_ferrite_aspect_ratio = float(line.split('x ')[1].strip())
        
        # Check if essential information is found and fill missing fields with 0
        ferrite_percentage = ferrite_percentage if ferrite_percentage is not None else 0.0
        martensite_percentage = martensite_percentage if martensite_percentage is not None else 0.0
        adjustment_martensite_size = adjustment_martensite_size if adjustment_martensite_size is not None else 0.0
        adjustment_martensite_aspect_ratio = adjustment_martensite_aspect_ratio if adjustment_martensite_aspect_ratio is not None else 0.0
        adjustment_ferrite_size = adjustment_ferrite_size if adjustment_ferrite_size is not None else 0.0
        adjustment_ferrite_aspect_ratio = adjustment_ferrite_aspect_ratio if adjustment_ferrite_aspect_ratio is not None else 0.0

        # Create the label array with 7 elements
        label_array = np.array([ferrite_percentage, martensite_percentage, num_bands,
                                adjustment_martensite_size, adjustment_martensite_aspect_ratio,
                                adjustment_ferrite_size, adjustment_ferrite_aspect_ratio])

        # Save the label array
        label_file_path = os.path.join(new_dir_path, 'label.npy')
        np.save(label_file_path, label_array)

        # Update the counts for number of bands
        if num_bands == 0:
            number_of_bands_zero += 1
        else:
            number_of_bands_more_than_zero += 1
        
        # Increase the counter of succesful processed directories
        counter += 1
    
    except (IndexError, ValueError) as e:
        print(f'Skipping {simulations_dir} due to error in Specs.txt: {e}')
        number_of_skipped_files += 1
    
    return counter

def perform_augmentation(data_dir: str) -> None:
    '''
    Enriches the dataset by introducing rotated versions of the existing data.

    Parameters
    ----------
    data_dir: str
        The path to the directory holding the data to be augmented.

    Returns
    ----------
    None
    '''
    # Iterate over directories
    for dirs in os.listdir(data_dir):
        sub_dir_path = os.path.join(data_dir, dirs)
        phase_file_path = os.path.join(sub_dir_path, 'phase_grid.npy')
        label_file_path = os.path.join(sub_dir_path, 'label.npy')
        
        if os.path.exists(phase_file_path) and os.path.exists(label_file_path):
            # Load phase array and label array
            phase_array = np.load(phase_file_path)
            label_array = np.load(label_file_path)

            # Check if the number of bands is more than 0
            num_bands = label_array[2]

            if num_bands >= 0:
                # Perform rotations
                for axis in range(3):
                    rotated_array = np.rot90(phase_array, k=1, axes=(axis, (axis + 1) % 3))

                    # Save the rotated array with a new name
                    rotated_file_path = os.path.join(sub_dir_path, f'phase_grid_rotated_{axis}.npy')
                    np.save(rotated_file_path, rotated_array)

                print(f"Augmented data for {sub_dir_path}")

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
        Number of succesfully processed directories.
    '''
    counter = 0
    for dir_name in sorted(os.listdir(input_dir)):
        full_dir_path = os.path.join(input_dir, dir_name)
        counter = process_simulation_dir(full_dir_path, output_dir, counter)
    return counter

# Traverse the directory structure starting from store_path_dir
traverse_directories(raw_data_dir, processed_data_dir)

# Perform augmentation
perform_augmentation(processed_data_dir)

print(number_of_skipped_files, "skipped due to missing files or Specs.txt errors.")
print(number_of_skipped_size, "skipped due to size.")
print(number_of_NOT_skipped, "processed successfully.")
print("Extracting phase process completed!")
print(number_of_bands_zero, "files with number of bands 0.")
print(number_of_bands_more_than_zero, "files with more than 0 bands.")