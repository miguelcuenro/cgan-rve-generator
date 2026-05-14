import numpy as np
import yaml
import os
import torch

# Load the parameters file
with open('parameters.yaml', 'r') as dictionary:
    parameters = yaml.safe_load(dictionary)

dataroot = os.path.expanduser(parameters['dataroot'])
img_size = parameters['img_size']
num_channels = parameters['num_channels']

def find_npy_files(root_dir):
    """
    Recursively find all files that match phase_grid*.npy.
    """
    npy_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file == 'phase_grid.npy':
                # if file.startswith('phase_grid') and file.endswith('.npy'): for the augmented files to get loaded
                npy_files.append(os.path.join(root, file))

    return npy_files


def load_data(npy_files):
    data_list = []
    label_list = []
    sample_ids = []
    label_shapes = set()  # Track unique label shapes

    for file_path in npy_files:
        npy_data = np.load(file_path, allow_pickle=True)  # Allow loading pickled data

        if npy_data.size == img_size ** 3:
            reshaped_array = npy_data.reshape(1, num_channels, img_size, img_size, img_size)
            data_list.append(reshaped_array)

            # Find the corresponding label.npy file
            label_path = os.path.join(os.path.dirname(file_path), 'label.npy')
            if os.path.exists(label_path):
                label_data = np.load(label_path, allow_pickle=True)
                label_shapes.add(label_data.shape)
                label_list.append(label_data)
                sample_id = os.path.basename(os.path.dirname(file_path))
                sample_ids.append(sample_id)
            else:
                print(f"Label file not found for {file_path}, skipping this file.")
                data_list.pop()  # Remove the last added grid as it doesn't have a label

    if not data_list:
        raise ValueError("No valid data found.")

    # Print the unique label shapes to debug the inconsistency
    print(f"Unique label shapes found: {label_shapes}")

    # Ensure all labels have the same shape
    if len(label_shapes) > 1:
        for label in label_list:
            print(f"Label shape: {label.shape}")
        raise ValueError("Inconsistent label shapes found.")

    data_np = np.concatenate(data_list, axis=0)
    labels_np = np.repeat(label_list, repeats=[data_np.shape[0] // len(label_list)], axis=0)

    return data_np, labels_np, sample_ids

import shutil  # optional, to get terminal width

def print_labels_table(label_list, sample_ids, label_names, max_label_len=9):
    """Print a table with truncated column headers to fit terminal."""
    if len(label_list) == 0:
        print("No labels to display.")
        return

    # Truncate each label name and prepend index
    headers = []
    for i, name in enumerate(label_names):
        short_name = name[:max_label_len] if len(name) > max_label_len else name
        headers.append(f"{i}. {short_name}")

    # Convert each label row to formatted strings
    data_rows = []
    for label_row in label_list:
        formatted_row = []
        for val in label_row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.2f}")
            else:
                formatted_row.append(str(val))
        data_rows.append(formatted_row)

    # Compute column widths (header vs data)
    num_cols = len(headers)
    col_widths = []
    for col in range(num_cols):
        max_header = len(headers[col])
        max_data = max((len(row[col]) for row in data_rows), default=0)
        col_widths.append(max(max_header, max_data) + 2)   # +2 for padding

    # Print header row
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print("\n" + header_line)
    print("-" * len(header_line))

    # Print data rows with sample ID
    for sid, row in zip(sample_ids, data_rows):
        cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        row_line = " | ".join(cells)
        print(f"{sid:<12} | {row_line}")

    print("\n")

npy_files = find_npy_files(dataroot)
data, label, id = load_data(npy_files)

labels_names = ["Ferrite percentage", "Martensite percentage", "Number of bands", "Adjustment of martensite size", "Adjustment of martensite aspect ratio", "Adjustment of ferrite size", "Adjustment of ferrite aspect ratio"]
# Note: In the cGAN training, only the first label (Ferrite percentage) is used for conditioning.

print_labels_table(label, id, labels_names)