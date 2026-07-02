import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import os
import sys
import datetime
from datetime import timedelta
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath('../utils'))
import visualization_utils as vu

class DCWCGANGP:
    def __init__(self,
                 directory, dataset,
                 batch_size, num_epochs, beta1, beta2, ngpu,learning_rate_disc, learning_rate_gen, d_loop,  # hparams training
                 lambda_penal, sigma,
                 img_size: int, num_channels,
                 gen_num_feature_maps, gen_dropout_rate, # particular gen init-parameters
                 dis_num_feature_maps, dis_dropout_rate,  # particular critic init-parameters
                 checkpoint_freq, checkpoint_step_freq
                 ):
        
        # Instantiate general (hyper)parameters
        self.directory = directory
        self.dataset = dataset  # RVEDataset = list of RVE-Objects

        self.batch_size = batch_size  # How many samples per batch
        self.num_epochs = num_epochs
        self.beta1 = beta1
        self.ngpu = ngpu
        self.learning_rate_disc = learning_rate_disc
        self.learning_rate_gen = learning_rate_gen
        self.d_loop = d_loop

        self.img_size = img_size
        self.num_of_z = int(img_size / 8)
        self.num_channels = num_channels
        self.lambda_penal = lambda_penal
        self.sigma = sigma

        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_step_freq = checkpoint_step_freq

        self.device = torch.device("cuda:0" if torch.cuda.is_available() & (self.ngpu >= 1) else "cpu")

        # Instantiate parts of the cGAN
        self.gen = Generator(num_channels=num_channels, num_feature_maps=gen_num_feature_maps, img_size=img_size, gen_dropout_rate=gen_dropout_rate).to(self.device).float()
        self.gen.apply(self.init_weights)

        self.critic = Critic(num_channels=num_channels, num_feature_maps=dis_num_feature_maps, img_size=img_size, dis_dropout_rate=dis_dropout_rate).to(self.device).float()
        self.critic.apply(self.init_weights)


        # Load data only if dataset is provided
        if dataset is not None:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        else:
            self.dataloader = None

        # Initialize storing variables
        self.fixed_label = torch.rand((self.batch_size, 1)).to(self.device).float() # Maybe we will have to restrain the range to [sample_min, sample_max] (let's see)
        # fixed_one_tensor = torch.ones(self.batch_size, 1, self.num_of_z, self.num_of_z, self.num_of_z)

        # fixed_label_expanded = fixed_label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # fixed_label_tensor = fixed_one_tensor * fixed_label_expanded

        self.fixed_z = torch.round(torch.randn(self.batch_size, self.num_channels, self.num_of_z, self.num_of_z, self.num_of_z, device=self.device).float())

        # self.fixed_noise = torch.cat([fixed_z, fixed_label_tensor], dim=1)

        self.G_losses = []
        self.D_losses = []
        self.gen_step_counter = 0
        self.dis_step_counter = 0


        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.critic.parameters(), lr=learning_rate_disc, betas=(beta1, beta2))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=learning_rate_gen, betas=(beta1, beta2))

        # Store parameters
        hparams = {
            'batch_size':batch_size,
            'num_epochs':num_epochs,
            'gen_num_feature_maps':gen_num_feature_maps,
            'gen_dropout_rate':gen_dropout_rate,
            'dis_num_feature_maps':dis_num_feature_maps,
            'dis_dropout_rate':dis_dropout_rate,
            'beta1':beta1,
            'beta2':beta2,
            'learning_rate_disc':learning_rate_disc,
            'learning_rate_gen':learning_rate_gen,
            'd_loop':d_loop,
            'lambda_penal':lambda_penal,
            'sigma':sigma,
            'img_size':img_size,
            'num_channels':num_channels,
            'checkpoint_freq': checkpoint_freq,
            'checkpoint_step_freq': checkpoint_step_freq
        }
        self.hparams = str(hparams)

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def compute_gradients(self, real_samples, fake_samples, labels):
        '''
        return: The gradient penalty for enforcing the lipschitz continuity
        '''

        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), self.num_channels, self.img_size, self.img_size, self.img_size,
                   device=self.device, dtype=real_samples.dtype)

         

        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.critic(interpolates, labels)

        fake = torch.full(size=(real_samples.shape[0], 1, 1, 1, 1), fill_value=1.0, device=self.device)
        fake.requires_grad = False

        # Get gradients w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        gradients = gradients[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(p=2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty

    def train(self, save_checkpoints=False, enable_sampling=False):
        # Raise error if no dataset is provided
        if self.dataloader is None:
            raise RuntimeError("Cannot train without a dataset. Provide a dataset in __init__.")
        
        print('\n' + "# ----------------- #" + '\n' + "Starting training..." + '\n' + "# ----------------- #")

        train_start = datetime.datetime.now()

        if hasattr(self, 'resume_log_dir') and self.resume_log_dir:
            training_log_dir = self.resume_log_dir
            timestamp = os.path.basename(training_log_dir)   # for sample filenames
            is_resuming = True

            # Clear the attribute so a subsequent train() call (without resume) creates a new dir
            self.resume_log_dir = None

        else:
            timestamp = train_start.strftime("%Y-%m-%d_%H-%M-%S")
            training_log_dir = os.path.join(os.path.expanduser("training_logs"), timestamp)
            is_resuming = False

        sample_dir = os.path.join(training_log_dir, "sample_dir")
        checkpoint_dir = os.path.join(training_log_dir, 'checkpoint_dir')

        if save_checkpoints == True or enable_sampling == True:
            writer = SummaryWriter(log_dir=training_log_dir)

        if save_checkpoints == True:
            # Create unique directories for sampling and logging
            #os.makedirs(training_log_dir, exist_ok=True)

            #sample_dir = os.path.join(training_log_dir, 'sample_dir') # I am under the impression we do not sample
            os.makedirs(checkpoint_dir, exist_ok=True)
            #os.makedirs(sample_dir, exist_ok=True)
            
            if not is_resuming:
                writer.add_text('Training Info', f'Start Time: {train_start.strftime('%Y-%m-%d %H:%M:%S')}', 0)
                writer.add_text('Hyperparameters', self.hparams, global_step=0)
            #writer.add_text('Special Comments', self.description, global_step=0)
        
        if enable_sampling == True:
            os.makedirs(sample_dir, exist_ok=True)

        sampling_freq = 150 # Again: Do we use this? Yes we do for SAMPLING #CHANGE IT TO 150
        sampling_counter = -1
        
        start_epoch = getattr(self, 'start_epoch', 0)  # default to 0 if not set
        for epoch in range(start_epoch, self.num_epochs):
            print('Epoch: ', epoch+1)
            for i, batch in tqdm(enumerate(self.dataloader), position=0):
                batch_data, batch_labels = batch  # Unpack the batch tuple
                self.gen_step_counter = epoch * len(self.dataloader) + i
                sampling_counter = sampling_counter + 1

                data = batch_data.to(self.device)
                labels = batch_labels[:, 0:1].to(self.device).float() # ensure compatible tensor operations during training or inference
                print("LABEL SIZE")
                print(labels.size())
                # Create random labeled input (NOT ANYMORE)
                #binary_noise = torch.randint(0, 2, (data.size(0), self.num_channels, self.num_of_z, self.num_of_z, self.num_of_z), device=self.device).double()
                noise = torch.rand(data.size(0), self.num_channels, self.num_of_z, self.num_of_z, self.num_of_z, device=self.device).float()
                # z_one_tensor = torch.ones(data.size(0), 1, self.num_of_z, self.num_of_z, self.num_of_z)
                # label_values = torch.rand((data.size(0), 1)) # Maybe we will have to restrain the range to [sample_min, sample_max] (let's see)

                # label_values_expanded = label_values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # label_tensor = one_tensor * label_values_expanded

                # Prepare real labels (same as later)
                # labels_expanded = labels[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Maybe it is better to normalize them from 0 to 1 (let's see)
                #label_tensor = z_one_tensor * labels_expanded

                # Concatenate input noise with real labels
                #z = torch.cat([noise, label_tensor], dim=1)

                fake_imgs = self.gen.forward(noise, labels)
                
                # Prepare fake samples for the critic
                # img_one_tensor = torch.ones(data.size(0), 1, self.img_size, self.img_size, self.img_size)
                # img_label_tensor = img_one_tensor * labels_expanded
                
                # critic_input_fake = torch.cat([fake_imgs, img_label_tensor], dim=1)

                if epoch == 51:
                    sampling_freq = 500 # Laut Xavi hier aufpassen (why???)

                # Sample the output of the generator
                if sampling_counter % sampling_freq == 0 and enable_sampling == True:
                    step = self.gen_step_counter

                    sample = fake_imgs[0].detach().cpu().numpy()
                    
                    filename = f'{step}th_sample_' + str(timestamp)
                    full_path = os.path.join(sample_dir, filename)
                    np.save(full_path, sample)

                    with torch.no_grad():
                        fake = self.gen.forward(self.fixed_z, self.fixed_label)[0].detach().cpu().numpy()

                        filename = f'{step}th_f_sample_' + str(timestamp)
                        full_path = os.path.join(sample_dir, filename)
                        np.save(full_path, fake)

                    vu.visualize_and_log_to_tensorboard(tag='Sample/z', tensor=fake, step=step, writer=writer)
                    vu.visualize_and_log_to_tensorboard(tag='Sample/fixed', tensor=fake, step=step, writer=writer)

                # Adapt the d-loop after some time
                if (epoch * len(self.dataloader) + i) > 5000:
                    self.d_loop = 5

                # Concatenate CRITIC input
                # labels_expanded = labels[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Maybe it is better to normalize them from 0 to 1 (let's see)
                # input_labels = img_one_tensor * labels_expanded
                # input_data = torch.cat([data, img_label_tensor], dim=1)
                
                # Training the DISCRIMINATOR
                for j in range(self.d_loop):
                    self.critic.zero_grad()

                    # Real data
                    outputs_real = self.critic.forward(data, labels)
                    d_loss_real = torch.mean(outputs_real)

                    # Fake data
                    outputs_fake = self.critic.forward(fake_imgs.detach(), labels)
                    d_loss_fake = torch.mean(outputs_fake)

                    # Save discriminator loss
                    d_loss = -(d_loss_real - d_loss_fake)
                    self.D_losses.append(d_loss)

                    gradient_penalty = self.compute_gradients(real_samples=data, fake_samples=fake_imgs.detach(), labels=labels) * self.lambda_penal

                    if save_checkpoints == True:
                        writer.add_scalar('Loss/D_real', d_loss_real.item(), global_step=self.dis_step_counter)
                        writer.add_scalar('Loss/D_fake', d_loss_fake.item(), global_step=self.dis_step_counter)
                        writer.add_scalar('Loss/Critic', d_loss.item(), global_step=self.dis_step_counter)
                        writer.add_scalar('GP', gradient_penalty.item(), global_step=self.dis_step_counter)
                    
                    total_d_loss = d_loss + gradient_penalty
                    total_d_loss.backward()
                    self.optimizerD.step()
                    
                    self.dis_step_counter += 1

                # Training the GENERATOR
                self.gen.zero_grad()

                outputs = self.critic.forward(fake_imgs, labels)
                
                # Save the generator's loss
                g_loss = -(torch.mean(outputs))
                self.G_losses.append(g_loss)

                statistical_penalty = 0

                if save_checkpoints == True:
                    writer.add_scalar('Loss/Generator', g_loss.item(), global_step=epoch*len(self.dataloader)+i)
                    writer.add_scalar('Loss/Statistical', 0, #statistical_penalty.item(),
                     global_step=epoch*len(self.dataloader)+i)
                
                total_g_loss = g_loss + statistical_penalty
                total_g_loss.backward()
                self.optimizerG.step()

            if save_checkpoints == True and ((epoch == self.num_epochs - 1) or (((epoch+1) % self.checkpoint_freq == 0) or (self.gen_step_counter % self.checkpoint_step_freq == 0))): # and (epoch != 0)) or (self.step_counter % 1000 == 0)) and save_checkpoints == True: # change epoch % 2 == 0 back to 100
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                checkpoint = {
                    'epoch': epoch + 1,
                    'generator_state_dict': self.gen.state_dict(),
                    'discriminator_state_dict':self.critic.state_dict(),
                    'generator_loss': self.G_losses[-1],
                    'discriminator_loss': self.D_losses[-1],
                    'optimizerG_state_dict': self.optimizerG.state_dict(),
                    'optimizerD_state_dict': self.optimizerD.state_dict(),
                    'gen_step_counter': self.gen_step_counter,
                    'dis_step_counter': self.dis_step_counter
                }
            
                torch.save(checkpoint, checkpoint_path)
                print(f'Checkpoint saved at epoch {epoch+1} to {checkpoint_path}')
        
        if save_checkpoints == True:
            train_end = datetime.datetime.now()
            duration = train_end - train_start
            duration_str = str(timedelta(seconds=duration.seconds))
            writer.add_text("Training Info", f"End Time: {train_end.strftime('%Y-%m-%d %H:%M:%S')}", 0)
            writer.add_text("Training Info", f"Duration: {duration_str}", 0)
            writer.add_scalar("Training Duration", duration.seconds, 0)

        if save_checkpoints == True or enable_sampling == True:
            writer.close()

        print('\n' + '# -------------- #' + '\n' + 'Training complete!' + '\n' + '# -------------- #')
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        print('\n' + '# -------------- #' + '\n' + 'Model loaded succesfully!' + '\n' + '# -------------- #')

        # Load the generator state dict and handle missing keys
        gen_state_dict = checkpoint['generator_state_dict']
        current_gen_state_dict = self.gen.state_dict()

        # Update the current state dict with the loaded state dict
        for name, param in gen_state_dict.items():
            if name in current_gen_state_dict:
                current_gen_state_dict[name].copy_(param)
            else:
                print(f"Skipping {name} as it is not in the current model state dict")

        # Load the updated state dict back into the model
        self.gen.load_state_dict(current_gen_state_dict)

        # Load the critic state dict and handle missing keys
        critic_state_dict = checkpoint['discriminator_state_dict']
        current_critic_state_dict = self.critic.state_dict()

        # Update the current state dict with the loaded state dict
        for name, param in critic_state_dict.items():
            if name in current_critic_state_dict:
                current_critic_state_dict[name].copy_(param)
            else:
                print(f"Skipping {name} as it is not in the current model state dict")

        # Load the updated state dict back into the model
        self.critic.load_state_dict(current_critic_state_dict)

        # Load the optimizers state dicts
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])

        # Last recorded epoch
        self.start_epoch = checkpoint.get('epoch', 0)

        # Derive the original log directory
        self.resume_log_dir = os.path.dirname(os.path.dirname(checkpoint_path))

        self.gen_step_counter = checkpoint.get('gen_step_counter', 0)
        self.dis_step_counter = checkpoint.get('dis_step_counter', 0)

        print(f"Resuming training from epoch {self.start_epoch}")
        print(f"Reusing log directory: {self.resume_log_dir}")

    def train_from(self, checkpoint_path, save_checkpoints=False, enable_sampling=False):
        self.load_checkpoint(checkpoint_path)
        self.train(save_checkpoints, enable_sampling)

class Critic(nn.Module):
    def __init__(self, num_channels, num_feature_maps, img_size, dis_dropout_rate):
        super().__init__()
        self.num_channels = num_channels
        self.num_feature_maps = num_feature_maps
        self.img_size = img_size
        self.dis_dropout_rate = dis_dropout_rate

        self.num_layers = int(np.log(img_size) / np.log(2))
        self.features = [2 ** i for i in range(self.num_layers - 2)]

        # 'Prelayer' - Makes the critic stronger
        self.prelayer = conv3d_same = nn.Conv3d(
            in_channels=num_channels+1,  # Number of input channels + number of label features
            out_channels=num_channels,  # Number of output channels
            kernel_size=3,  # 3x3x3 kernel
            stride=1,  # Stride of 1
            padding=1,  # Padding set to 1 to maintain dimensions
            bias=False  # Whether to use bias or not
        )

        # Initial layer
        self.initial_layer = nn.Conv3d(
            in_channels=num_channels,
            out_channels=self.num_feature_maps,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        # Hidden layers
        layers = []

        for i in range(self.num_layers - 3):
            i_in = num_feature_maps * self.features[i]
            i_out = num_feature_maps * self.features[i+1]
            i_kernel_size = 4
            i_stride = 2
            i_padding = 1

            layers.append(
                nn.Sequential(
                    nn.Conv3d(i_in, i_out, i_kernel_size, i_stride, i_padding, bias=False),
                    nn.BatchNorm3d(i_out),
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(self.dis_dropout_rate)
                )
            )
        
        # Transforms the List of Layers into ModuleList, so that PyTorch is aware of the modules (optimizes handling)
        self.hidden_layers = nn.ModuleList(layers)

        # Final layer
        self.final_layer = nn.Conv3d(  # 2x2x2 -> 1
            in_channels=num_feature_maps * self.features[-1], out_channels=1,
            kernel_size=4, stride=2, padding=0, bias=False
        )

        # Hidden layers to learn meaningful representation of the labels
        self.label_layers = nn.Sequential(
            nn.Linear(1, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 32768),
            nn.LeakyReLU(0.2)
        ).float()

    def forward(self, sample, label):
        # Cast to the same dtype as the critic's parameters
        sample = sample.to(next(self.parameters()).dtype)
        label = label.to(next(self.parameters()).dtype)

        # Compute label projection
        lp = self.label_layers(label)

        # Fold into a cube
        lp_3d = lp.view(-1, 1, 32, 32, 32)

        # Concatenate
        z = torch.cat([sample, lp_3d], dim=1)

        z = nn.LeakyReLU(0.2)(self.prelayer(z))
        z = nn.LeakyReLU(0.2)(self.initial_layer(z))
        for layer in self.hidden_layers:
            z = layer(z)
        z = self.final_layer(z)
        return z

class Generator(nn.Module):
    def __init__(self, num_channels, num_feature_maps, img_size, gen_dropout_rate):
        super().__init__()
        self.num_channels = num_channels
        self.num_feature_maps = num_feature_maps
        self.img_size = img_size
        self.dropout_rate = gen_dropout_rate

        self.num_layers = int(np.log(img_size) / np.log(2))
        self.features = [2 ** i for i in range(self.num_layers - 2)]
        self.features.reverse()

        self.padding = nn.ReflectionPad3d(1)

        # Initial layer
        self.initial_layer = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=2, out_channels=num_feature_maps//self.features[0],
                kernel_size=4, stride=2, padding=4, bias=False
            ), # in_channels=2 --> noise and label
            nn.BatchNorm3d(num_feature_maps//self.features[0])
        )

        # Hidden layers
        layers = []
        for i in range(self.num_layers-3):
            i_in = num_feature_maps // self.features[i]
            i_out = num_feature_maps // self.features[i+1]
            i_kernel_size = 4
            i_stride = 2
            i_padding = 2

            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(i_in, i_out, i_kernel_size, i_stride, i_padding, bias=False),
                    nn.BatchNorm3d(i_out),
                    nn.LeakyReLU(0.2),
                    nn.Dropout3d(self.dropout_rate)
                )
            )
        
        # Transforms the List of Layers into ModuleList, so that PyTorch is aware of the modules (optimizes handling)
        self.hidden_layers = nn.ModuleList(layers)

        # Final layer
        self.final_layer = nn.ConvTranspose3d(  # 32x32x32 -> 64x64x64, c=1
            in_channels=i_out,  # Corrected in_channels
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            bias=True #False
        )

        # Hidden layers to learn meaningful representation of the labels
        self.label_layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2)
        ).float()

    def forward(self, noise, label):
        # Compute label projection
        lp = self.label_layers(label)

        # Fold into a cube
        lp_3d = lp.view(-1, 1, 4, 4, 4)

        # Concatenate
        z = torch.cat([noise, lp_3d], dim=1)

        # Train the model
        z = self.padding(z) # 6x6x6
        z = self.initial_layer(z) # 6x6x6
        z = nn.LeakyReLU(0.2)(z)
        z = self.padding(z) # 8x8x8
        for layer in self.hidden_layers: # Happens twice
            z = nn.LeakyReLU(0.2)(layer(z))
            z = self.padding(z)
        z = self.final_layer(z) # nn.LeakyReLU(0.2)(self.final_layer(z)) # 30x30x30 
        z = self.padding(z) # batch_sizex1x32x32x32
        z = torch.sigmoid(z)
        return z

######################
# For testing purposes
######################
from torch.utils.data import Dataset

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

        if npy_data.size == img_size ** 3:
            reshaped_array = npy_data.reshape(1, num_channels, img_size, img_size, img_size)
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

    data_tensor = torch.from_numpy(data_np).float()
    labels_tensor = torch.from_numpy(labels_np).float()

    return CustomDataset(data_tensor, labels_tensor)

if __name__ == "__main__":
    # Path to root directory
    dataroot = os.path.expanduser("../../data/processed_data")
    # Number of simultaneous loads of data into the RAM
    num_of_workers = 1
    # Batch size during training
    batch_size = 8  # TODO should this be increased?, # TODO2 research and experiment
    # Spatial size of the training volumes. Every volume will be resized to this
    img_size = 32
    # Number of channels in the training samples, is 1 here because we only look at orientation, but could be increased when
    # we add phases etc.
    num_channels = 1
    # size of the latent noise vector is now computed based on img_size: num_of_z = 8, for 64 cubes

    # Number of feature maps in the generator
    gen_num_feature_maps = 16  # was 32
    gen_dropout_rate = 0
    # Number of feature maps in the discriminator
    dis_num_feature_maps = 8  # was 16
    dis_dropout_rate = 0
    # Number of training epochs
    num_epochs = 2  # TODO2 research and experiment

    # learning rate for the optimizers
    learning_rate_disc = 0.0002
    learning_rate_gen = 0.00008
    # factor which decides how many times the critic is trained for each gen training step
    d_loop = 10
    # beta1 hyperparameter for the Adam optimizer
    beta1 = 0.65
    beta2 = 0.8
    # number of available gpus, 0 for cpu mode
    ngpu = 1
    # lambda multiplier for the gradient penalty
    lambda_penal = 10
    sigma = (10 ** 3)  # maybe dynamic?
    
    # ------------ #
    # PREPARE DATA #
    # ------------ #

    npy_files = find_npy_files(dataroot)
    dataset = load_data(npy_files)

    # ----------------------- #
    # CREATE GAN AND TRAIN IT #
    # ----------------------- #
    cgan = DCWCGANGP(dataroot, dataset,
                        batch_size, num_epochs, beta1, beta2, ngpu, learning_rate_disc, learning_rate_gen, d_loop,
                        # hyperparameters for training
                        lambda_penal, sigma,
                        img_size, num_channels,  # size parameters
                        gen_num_feature_maps, gen_dropout_rate,  # particular gen init-parameters
                        dis_num_feature_maps, dis_dropout_rate,
                        checkpoint_freq=100, checkpoint_step_freq=1000)  # particular critic init-parameters)

    cgan.train(save_checkpoints=False, enable_sampling=False)