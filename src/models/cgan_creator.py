import torch
import yaml
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cGAN
import pyvista as pv

torch.cuda.empty_cache()  # Wegen Grafikspeicher auf dem Cluster
pv.start_xvfb()  # für Screenshot auf dem Cluster

# --------------- #
# SETUP WORKFRAME #
# --------------- #

# Define custom class
class CustomDataset(Dataset):
    def __init__(self, data, labels):  # , labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def find_npy_files(root_dir):
    """
    Recursively find all files that match phase_grid*.npy.
    """
    npy_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'phase_grid.npy':
                # if file.startswith('phase_grid') and file.endswith('.npy'): for the augmented files to get loaded
                npy_files.append(os.path.join(root, file))

    return npy_files


def load_data(npy_files):
    data_list = []
    label_list = []
    label_shapes = set()  # Track unique label shapes

    for file_path in npy_files:
        npy_data = np.load(file_path, allow_pickle=True)  # Allow loading pickled data

        if npy_data.size == image_size ** 3:
            reshaped_array = npy_data.reshape(1, num_channels, image_size, image_size, image_size)
            data_list.append(reshaped_array)

            # Find the corresponding label.npy file
            label_path = os.path.join(os.path.dirname(file_path), 'label.npy')
            if os.path.exists(label_path):
                label_data = np.load(label_path, allow_pickle=True)
                label_shapes.add(label_data.shape)
                label_list.append(label_data)
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

    data_tensor = torch.from_numpy(data_np).double()
    labels_tensor = torch.from_numpy(labels_np).double()

    return CustomDataset(data_tensor, labels_tensor)

# ---------------- #
# HYPER PARAMETERS #
# ---------------- #

# Load the parameters file
with open('parameters.yaml', 'r') as dictionary:
    parameters = yaml.safe_load(dictionary)

# Path to root directory
dataroot = os.path.expanduser(parameters['dataroot'])
num_of_workers = parameters['num_of_workers']
batch_size = parameters['batch_size']
image_size = parameters['img_size']
num_channels = parameters['num_channels']
gen_num_feature_maps = parameters['gen_num_feature_maps']
gen_dropout_rate = parameters['gen_dropout_rate']
dis_num_feature_maps = parameters['dis_num_feature_maps'] # was 16
dis_dropout_rate = parameters['dis_dropout_rate']
num_epochs = parameters['num_epochs']
learning_rate_disc = parameters['learning_rate_disc']
learning_rate_gen = parameters['learning_rate_gen']
d_loop = parameters['d_loop']
beta1 = parameters['beta1']
beta2 = parameters['beta2']
ngpu = parameters['ngpu']
lambda_penal = parameters['lambda_penal']
sigma = parameters['sigma']

# ------------ #
# PREPARE DATA #
# ------------ #

npy_files = find_npy_files(dataroot)
dataset = load_data(npy_files)

# ----------------------- #
# CREATE GAN AND TRAIN IT #
# ----------------------- #
cgan = cGAN.DCWCGANGP(dataroot, dataset,
                        batch_size, num_epochs, beta1, beta2, ngpu, learning_rate_disc, learning_rate_gen, d_loop,
                        # hyperparameters for training
                        lambda_penal, sigma,
                        image_size, num_channels,  # size parameters
                        gen_num_feature_maps, gen_dropout_rate,  # particular gen init-parameters
                        dis_num_feature_maps, dis_dropout_rate)  # particular critic init-parameters)

if parameters['from_checkpoint'] == False:
    cgan.train(save_checkpoints=parameters['save_checkpoints'], enable_sampling=parameters['enable_sampling'])
else:
    cgan.train_from(os.path.expanduser(parameters['checkpoint']))