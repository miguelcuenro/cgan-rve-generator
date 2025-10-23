import cGAN as gan_sourcecode
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

torch.cuda.empty_cache()

# --------------- #
# SETUP WORKFRAME #
# --------------- #

# Define custom class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
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

def log_memory_usage(step):
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Step {step}: Allocated memory: {allocated_memory} bytes, Reserved memory: {reserved_memory} bytes")

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
img_size = parameters['img_size']
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

checkpoint = parameters['checkpoint']

operated_cgan = cGAN.DCWCGANGP(directory=dataroot,
                                    dataset=dataset,
                                    batch_size=batch_size, num_epochs=num_epochs, beta1=beta1, beta2=beta2, ngpu=ngpu,
                                    learning_rate_disc=learning_rate_disc, learning_rate_gen=learning_rate_gen,
                                    d_loop=d_loop, lambda_penal=lambda_penal, sigma=sigma,
                                    img_size=img_size, num_channels=num_channels,
                                    gen_num_feature_maps=gen_num_feature_maps, gen_dropout_rate=gen_dropout_rate,
                                    dis_num_feature_maps=dis_num_feature_maps, dis_dropout_rate=dis_dropout_rate)

print(len(dataset))
print(operated_cgan.device)
print(operated_cgan.description)

torch.cuda.empty_cache()

# Load the checkpoint
operated_cgan.load_checkpoint(checkpoint_path=checkpoint)
print("Loaded the checkpoint!")

# Perform analysis with memory management
torch.cuda.empty_cache()

# Setup the sampling directory
root = os.path.join(parameters['root'], datetime.datetime.now()strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(root, exist_ok=True)

# Z IS WROOOOONG!!! CHANGE IT MIGUELON
number_of_samples = parameters['number_of_samples']

for i in range(n):
    binary_noise = torch.randn(operated_cgan.batch_size, operated_cgan.num_channels, operated_cgan.num_of_z, operated_cgan.num_of_z,
                        operated_cgan.num_of_z, device=operated_cgan.device).double()
    
    label_values = torch.rand((operated_cgan.batch_size, 1))
    one_tensor = torch.ones(operated_cgan.batch_size, 1, operated_cgan.num_of_z, operated_cgan.num_of_z, operated_cgan.num_of_z)

    label_values_expanded = label_values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    label_tensor = one_tensor * label_values_expanded

    z = torch.cat([binary_noise, label_tensor], dim=1)

    generated_img_raw = operated_cgan.gen(z)
    generated_img_clean = torch.round(generated_img_raw)
    filename = f"number_{i}"
    save_path = os.path.join(root, filename)
    np.save(save_path, generated_img_clean[0].detach().numpy())