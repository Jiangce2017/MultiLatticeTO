import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
#from Data_Plotting.Shape_Plotting import CustomDataset
from torcheval.metrics import R2Score, Mean
import pickle
from tqdm import tqdm
import time
import itertools
from sklearn.preprocessing import MinMaxScaler
########################################################################################################################



########################################################################################################################
# Class to normalize a dataset between [0,1]
class PytorchMinMaxScaler:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, data):
        self.min_vals, self.max_vals = torch.min(data, dim=0)[0], torch.max(data, dim=0)[0]

    def transform(self, data):
        # Check if the scaler has been fitted
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform()")

        # Flatten and normalize the data
        flattened_data = self._flatten(data)
        normalized_data = self._normalize(flattened_data)
        normalized_reshaped_data = self._reshape(normalized_data, data.shape)
        return normalized_reshaped_data

    def inverse_transform(self, scaled_data):
        # Check if the scaler has been fitted
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Scaler has not been fitted. Call fit() before inverse_transform()")

        # Flatten and normalize the data
        flattened_data = self._flatten(scaled_data)

        # Inverse transform
        original_data = self._inverse_transform(flattened_data)

        # Reshape the data to its original shape
        reshaped_data = self._reshape(original_data, scaled_data.shape)

        return reshaped_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _flatten(self, data):
        return data.view(len(data), -1)

    def _normalize(self, flattened_data):
        # Ensure that min_vals and max_vals are on the same device as flattened_data
        min_vals = self.min_vals.to(flattened_data.device)
        max_vals = self.max_vals.to(flattened_data.device)

        # Perform normalization
        normalized_data = (flattened_data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def _inverse_transform(self, scaled_data):
        # Ensure that min_vals and max_vals are on the same device as flattened_data
        min_vals = self.min_vals.to(scaled_data.device)
        max_vals = self.max_vals.to(scaled_data.device)

        return scaled_data * (max_vals - min_vals) + min_vals

    def _reshape(self, data, original_shape):
        return data.view(original_shape)


# Create a function to check if a data point is binary
def is_binary(tensor):
    unique_values = torch.unique(tensor)
    return set(unique_values.tolist()) == {0, 1}
########################################################################################################################



########################################################################################################################
# Create the classes for the model

# Class to compute the latent points based on distribution variance and mean
class SampleLatentFeatures(nn.Module):
    def __init__(self):
        super(SampleLatentFeatures, self).__init__()

    def forward(self, distribution_mean, distribution_variance):
        # compute the batch size
        batch_size = distribution_variance.shape[0]

        # Returns a tensor filled with random numbers from a normal distribution
        random = torch.randn((batch_size, distribution_variance.shape[1]), device=distribution_mean.device)
        return distribution_mean + torch.exp(0.5 * distribution_variance) * random


# Encoder Class
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        # Geometry Layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32, 16)

        self.mean = nn.Linear(28, latent_dims)
        self.variance = nn.Linear(28, latent_dims)
        self.relu = nn.ReLU()

        # self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        self.kl = 0

        # Property 1 Layers
        self.conv4 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 9)

        # Property 2 Layers
        self.dense4 = nn.Linear(1, 64)
        self.dense5 = nn.Linear(64, 3)

        # Import the sampling class
        self.sample_latent_layer = SampleLatentFeatures()

    def forward(self, x, y, v, scaler):
        # Normalize property 1 Data
        y = scaler.transform(y)

        # reshape to account for batch size:
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        y = y.view(y.shape[0], 1, y.shape[1], y.shape[2])
        v = v.view(v.shape[0], 1)

        # Geometry
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x.view(-1, 32))

        # Property 1
        y = self.conv4(y)
        y = self.relu(y)
        y = self.maxpool2(y)
        y = self.conv5(y)
        y = self.relu(y)
        y = y.view((-1, 64))
        y = self.dense2(y)
        y = self.dense3(y)

        # Property 2
        v = self.dense4(v)
        v = self.dense5(v)

        # Combine all the data into a single vector
        concat = torch.cat((x, y, v), 1)

        # Compute the mean and variance of the combined vector
        distribution_mean = self.mean(concat)
        distribution_variance = self.variance(concat)

        # Return the latent encoding of the features
        latent_encoding = self.sample_latent_layer(distribution_mean, distribution_variance)

        # Compute and Update the KL-divergence
        self.kl = -0.5 * torch.mean(1 + distribution_variance - (distribution_mean ** 2) - torch.exp(distribution_variance))

        return latent_encoding


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()

        # Geometry Layers
        self.dense1 = nn.Linear(latent_dims, 64)
        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1)
        self.upsample1 = nn.Upsample(scale_factor=(2, 2))
        self.conv3 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=1)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2))
        self.conv4 = nn.ConvTranspose2d(64, 1, kernel_size=(5, 5), stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Property 1 Layers
        self.dense2 = nn.Linear(latent_dims, 128)
        self.dense3 = nn.Linear(128, 64)
        self.upsample3 = nn.Upsample(scale_factor=(3, 3))
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=1, padding=1)

        # Property 2 Layers
        self.dense4 = nn.Linear(latent_dims, 64)
        self.dense5 = nn.Linear(64, 1)

    def forward(self, latent_embedding, scaler):
        # Geometry Decoding
        x = self.dense1(latent_embedding)
        x = x.view((-1, 64, 1, 1))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.upsample1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.upsample2(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        x = x.view((-1, 28, 28))

        # Property 1
        y = self.dense2(latent_embedding)
        y = self.dense3(y)
        y = y.view((-1, 64, 1, 1))
        y = self.conv5(y)
        y = self.relu(y)
        y = self.upsample3(y)
        y = self.conv6(y)
        y = self.sigmoid(y)
        y = y.view((-1, 3, 3))

        # Property 2
        v = self.dense4(latent_embedding)
        v = self.dense5(v)
        v = self.sigmoid(v)
        v = v.view(-1)

        # Return the stiffness de-normalized
        return x, scaler.inverse_transform(y), v


# VAE Architecture
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, y, v, scaler):
        z = self.encoder(x, y, v, scaler)
        return self.decoder(z, scaler)



########################################################################################################################
# Set the Callbacks for the model


# This class will end the training if the loss does not improve by min_delta over a set patience(# epochs)
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# This class is to save the model's best weights
class ModelCheckpoint:
    def __init__(self, model, scaler, save_path):
        self.model = model
        self.scaler = scaler
        self.save_path = save_path
        self.best_loss = float('inf')

    def update(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            # Save model
            with open(self.save_path, "wb") as f:
                pickle.dump({'model': self.model.state_dict(), 'scaler': self.scaler}, f)

            return True

        else:
            return False


########################################################################################################################
# Define the training process for the VAE
def train(autoencoder, train, test, num_epochs, patience=10, min_delta=0):
    # Set the optimizer parameters
    opt = torch.optim.Adam(autoencoder.parameters(), eps=1e-07)

    # Get the total number of samples in the dataset
    ntrain = len(train.dataset)

    # Check if the data is binary
    data_samples = []
    # Check the first 10 samples
    for i in range(10):
        data_samples.append(torch.tensor(train.dataset[i][0]))

    # Convert the list of tensors to a stacked tensor along a new dimension
    data_samples = torch.stack(data_samples, dim=0)

    # Check if the geometry is binary
    binary_boolean = is_binary(data_samples)

    if binary_boolean:
        print("Your geometry data has been classified as binary, if this is not correct then modify binary_boolean.")
    else:
        print("Your geometry data has been classified as continuous, if this is not correct then modify binary_boolean.")

    # Get the batch size from the DataLoader
    batch_size = train.batch_size
    # Set up the scheduler
    iterations = epochs * (ntrain // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iterations)

    # Initialize lists to store the R^2 and loss values for the entirety of training
    r2_plot = []
    r2_plot_validation = []
    r2_plot_property_1 = []
    r2_plot_validation_property_1 = []
    r2_plot_property_2 = []
    r2_plot_validation_property_2 = []
    loss_plot = []
    loss_plot_validation = []

    reconstruction_plot = []
    kl_plot = []

    # Initialize the metrics to track the R^2 and losses of each batch
    r2 = R2Score()
    r2_validation = R2Score()

    r2_property_1 = R2Score()
    r2_validation_property_1 = R2Score()

    r2_property_2 = R2Score().to(device)
    r2_validation_property_2 = R2Score().to(device)

    losses = Mean().to(device)
    losses_validation = Mean().to(device)

    # Other metrics to track for geometry
    reconstruction_average = Mean().to(device)
    kl_divergence_average = Mean().to(device)

    # Initialize for the plot
    best_epoch = 0

    # Record the total training time
    total_training_time = 0.0

    # Initialize the Callbacks
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    checkpoint = ModelCheckpoint(autoencoder, stiffness_scaler, 'vae.pickle')

    # Iterate through each epoch
    for epoch in range(num_epochs):
        # Start the timer for the current epoch
        epoch_start_time = time.time()

        # Initialize a tqdm progress bar
        progress_bar = tqdm(enumerate(train), total=len(train))

        # Iterate through all the batches
        for batch_idx, batch_data in progress_bar:
            # x is a tensor containing all the arrays, y is a tensor containing all the stiffness tensors, v contains the volume fractions
            x, y, v = batch_data

            # Load the batch to the GPU
            x = x.to(device)
            y = y.to(device)
            v = v.to(device)

            # Initialize the gradients
            opt.zero_grad()

            # Predict the batch
            x_hat, y_hat, v_hat = autoencoder(x, y, v, stiffness_scaler)
            #
            # if epoch>9:
            #     if batch_idx == 0:
            #         plt.imshow(x[0].view(28, 28).cpu().detach().numpy(), vmin=0, vmax=1.0)
            #         plt.show()
            #         plt.imshow(x_hat[0].view(28, 28).cpu().detach().numpy(), vmin=0, vmax=1.0)
            #         plt.show()

            # Compute the Batch Size
            batch_size = torch.tensor(x.shape[0]).to(device)

            # Update the progress bar description
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train)}")

            # Calculate MSE and then multiply by the shape to scale it properly
            # Reconstruction loss is the difference between the true data and the reconstructed data
            if binary_boolean:  # binary cross-entropy for binary data
                geometry_reconstruction_loss = torch.mean(
                    torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')) * x.shape[1] * x.shape[2]
            else:  # MSE loss for continuous data
                geometry_reconstruction_loss = torch.mean(
                    torch.nn.functional.mse_loss(x_hat, x, reduction='mean')) * \
                                               x.shape[1] * x.shape[2]
            property_1_reconstruction_loss = torch.mean(torch.nn.functional.mse_loss(stiffness_scaler.transform(y_hat), stiffness_scaler.transform(y), reduction='mean')) * y.shape[1] * y.shape[2]
            property_2_reconstruction_loss = torch.mean(torch.nn.functional.mse_loss(v_hat, v, reduction='mean'))

            # Combine the Losses
            reconstruction_loss = geometry_reconstruction_loss + property_1_reconstruction_loss + property_2_reconstruction_loss

            # Loss is the sum of the reconstructed loss and KL-divergence
            loss = reconstruction_loss + autoencoder.encoder.kl

            # Calculate the R^2 value for the batch
            r2.update(x_hat.reshape(batch_size, -1), x.reshape(batch_size, -1))
            r2_property_1.update(y_hat.reshape(batch_size, -1), y.reshape(batch_size, -1))
            r2_property_2.update(v_hat, v)
            losses.update(loss.detach(), weight=batch_size)

            # Record the average reconstruction loss and kl-divergence for the batch
            reconstruction_average.update(reconstruction_loss.detach(), weight=batch_size)
            kl_divergence_average.update(autoencoder.encoder.kl.detach(), weight=batch_size)

            loss.backward()  # Computes the gradients
            opt.step()  # Updates the weights
            scheduler.step()

        # Calculate the training time for the epoch
        training_time = time.time() - epoch_start_time
        total_training_time += training_time
        '''
        # Test print the parameters of the architecture
        print(list(autoencoder.parameters()))
        for name, param in autoencoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"Parameter name: {name}")
                print(f"Gradient: {param.grad}")
        '''
        # Record the loss values for the Epoch
        loss_plot.append(losses.compute())
        losses.reset()

        # print(geometry_reconstruction_loss, r2.compute())

        # Compute the R^2 values for the Epoch and add to list
        r2_plot.append(r2.compute())
        r2_plot_property_1.append(r2_property_1.compute())
        r2_plot_property_2.append(r2_property_2.compute())
        r2.reset()
        r2_property_1.reset()
        r2_property_2.reset()

        # Compute the average reconstruction loss and kl-divergence for the Epoch and add to list
        reconstruction_plot.append(reconstruction_average.compute())
        kl_plot.append(kl_divergence_average.compute())
        reconstruction_average.reset()
        kl_divergence_average.reset()

        with torch.no_grad():
            for x, y, v in test:
                # x is a tensor containing all the arrays, y is a tensor containing all the stiffness tensors, v contains the volume fractions
                # Load the batch to the GPU
                x = x.to(device)
                y = y.to(device)
                v = v.to(device)

                # Predict the batch
                x_hat, y_hat, v_hat = autoencoder(x, y, v, stiffness_scaler)

                # Compute the Batch Size
                batch_size = x.shape[0]

                # Calculate MSE and then multiply by the shape to scale it properly
                # Reconstruction loss is the difference between the true data and the reconstructed data
                if binary_boolean:  # binary cross-entropy for binary data
                    geometry_reconstruction_loss = torch.mean(
                        torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')) * x.shape[1] * x.shape[2]
                else:  # MSE loss for continuous data
                    geometry_reconstruction_loss = torch.mean(
                        torch.nn.functional.mse_loss(x_hat, x, reduction='mean')) * \
                                                   x.shape[1] * x.shape[2]
                property_1_reconstruction_loss = torch.mean(torch.nn.functional.mse_loss(stiffness_scaler.transform(y_hat), stiffness_scaler.transform(y), reduction='mean')) * y.shape[1] * y.shape[2]
                property_2_reconstruction_loss = torch.mean(torch.nn.functional.mse_loss(v_hat, v, reduction='mean'))

                # Combine the Losses
                reconstruction_loss = geometry_reconstruction_loss + property_1_reconstruction_loss + property_2_reconstruction_loss

                # Loss is the sum of the reconstructed loss and KL-divergence
                loss_validation = reconstruction_loss + autoencoder.encoder.kl

                # Calculate the R^2 value for the batch
                r2_validation.update(x_hat.reshape(batch_size, -1), x.reshape(batch_size, -1))
                r2_validation_property_1.update(y_hat.reshape(batch_size, -1), y.reshape(batch_size, -1))
                r2_validation_property_2.update(v_hat.reshape(batch_size), v.reshape(batch_size))
                losses_validation.update(loss_validation.detach(), weight=batch_size)

        # Record the loss values for the Epoch
        batch_loss_validation = losses_validation.compute()
        loss_plot_validation.append(batch_loss_validation)

        # Update the checkpoint if the validation loss is better
        if checkpoint.update(batch_loss_validation):
            best_epoch = epoch

        # Record the R^2 value for the Epoch
        r2_plot_validation.append(r2_validation.compute())
        r2_plot_validation_property_1.append(r2_validation_property_1.compute())
        r2_plot_validation_property_2.append(r2_validation_property_2.compute())

        # Check the early stopping condition
        if early_stopper.early_stop(batch_loss_validation):
            break

        # Reset the Recording Metrics for the next Epoch
        losses_validation.reset()
        r2_validation.reset()
        r2_validation_property_1.reset()
        r2_validation_property_2.reset()

    # Plot the average reconstruction and kl-divergence vs the epoch
    plt.plot(torch.stack(reconstruction_plot).cpu().detach().numpy(), c='red', label='reconstruction loss')
    plt.plot(torch.stack(kl_plot).cpu().detach().numpy(), c='blue', label='kl-divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig("average_reconstruction_kl_divergence", bbox_inches='tight')
    plt.close()

    # Plot the coefficient of determination of each variable through training
    list_of_r2s = [r2_plot, r2_plot_validation, r2_plot_property_1, r2_plot_validation_property_1, r2_plot_property_2, r2_plot_validation_property_2]
    r2_labels = ["Geometry", "Geometry Validation", "Stiffness", "Stiffness Validation", "Volume Fraction", "Volume Fraction Validation"]
    plot_r2s(list_of_r2s, r2_labels, total_training_time, latent_dims, num_epochs, best_epoch)

    # Plot the average coefficient of determination and loss through training
    plot_metrics(r2_plot, r2_plot_validation, loss_plot, loss_plot_validation, total_training_time, latent_dims, num_epochs, best_epoch)

    return autoencoder


def plot_r2s(list_r2s, labels, time, latent_dimensions, num_epochs, best_epoch):
    # list_r2s should have train, validation sets for 3 variables

    # Initialize the Figure
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Coefficient of Determination', fontsize=14)

    plt.title("Latent Space Dimensionality: " + str(latent_dimensions) +
              "\nTotal Epochs: " + str(len(list_r2s)) +
              "\nTotal Time: " + str(time) +
              "\nBest Epoch: " + str(best_epoch))

    # Define the colors for the plots
    colors = ['crimson', 'pink', 'indigo', 'darkviolet', 'navy', 'slateblue']

    # Plot each of the coefficients of determination
    for r2, label, color in zip(list_r2s, labels, colors):
        r2 = torch.stack(r2).cpu().detach().numpy()
        ax.plot(r2, label=label, color=color)

    # Bound the axes so that they are comparable to other plots
    plt.ylim(0, 1.1)
    plt.xlim(0, num_epochs)

    # Set a horizontal line at 95% reconstruction performance
    ax.axhline(y=0.95, color='r', linestyle='-', label="95% Coefficient of Determination")
    plt.yticks(fontsize=14)

    fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("all_coefficients_determination", bbox_inches='tight')
    # plt.show()
    plt.close()
# Plot the metrics from training:
def plot_metrics(r2, r2_validation, loss, loss_validation, time, latent_dimensions, num_epochs, best_epoch):
    # convert to numpy and detach from the gpu
    r2 = torch.stack(r2).cpu().detach().numpy()
    loss = torch.stack(loss).cpu().detach().numpy()
    best_r2 = max(r2)

    r2_validation = torch.stack(r2_validation).cpu().detach().numpy()
    loss_validation = torch.stack(loss_validation).cpu().detach().numpy()
    best_r2_validation = max(r2_validation)

    # Initialize the Figure
    fig, ax1 = plt.subplots()
    plt.title("Latent Space Dimensionality: " + str(latent_dimensions) + "\nBest Training R^2: " + str(best_r2) +
              "\nBest Validation R^2: " + str(best_r2_validation) +
              "\nTotal Epochs: " + str(len(r2)) +
              "\nTotal Time: " + str(time) +
              "\nBest Epoch: " + str(best_epoch))
    # Set the axes for Loss
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)

    # Bound the axes so that they are comparable to other plots
    plt.xlim(0, num_epochs)
    plt.ylim(0, 150)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.plot(loss, label="Training Loss", color='blue')
    ax1.plot(loss_validation, label="Validation Loss", color='orange')

    # Set the axes for Coefficient of Determination
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Coefficient of Determination', fontsize=14)  # we already handled the x-label with ax1
    plt.ylim(0, 1.1)
    ax2.plot(r2, label="Training Coefficient of Determination", color='cornflowerblue')
    ax2.plot(r2_validation, label="Validation Coefficient of Determination", color='moccasin')
    ax2.axhline(y=0.95, color='r', linestyle='-', label="95% Coefficient of Determination")
    plt.yticks(fontsize=14)

    fig.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("hybrid_loss_coefficient_determinination", bbox_inches='tight')
    # plt.show()
    plt.close()

# ########################################################################################################################
# # Train the Model
# if __name__ == "__main__":  # Only runs training in this script
#     # Define the parameters of the Data
#     """
#     Image size
#     Dataset size
#     """

#     # Define the parameters of the Autoencoder
#     latent_dims = 16
#     image_size = 28
#     epochs = 300
#     batch_size = 32

#     # Set the device for training
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     ########################################################################################################################
#     # Load the Dataset
#     dataset = CustomDataset('Combined_Space_1_density.csv')
#     print("data", dataset)
#     # Load the data needed for training

#     # Load all the datapoints to get all the stiffness values
#     for idx in range(len(dataset)):
#         _ = dataset[idx]  # Call __getitem

#     # Access the stiffness data from the dataset
#     stiffness_data = dataset.get_all_stiffness_values()

#     # Create a scaler to normalize the data from [0,1]
#     stiffness_scaler = PytorchMinMaxScaler()
#     stiffness_scaled = stiffness_scaler.fit_transform(torch.tensor(stiffness_data))

#     # Split the data into training and validation data
#     train_set_original, test_set_original = torch.utils.data.random_split(dataset, [0.85, 0.15])

#     # Create a dictionary to store the split data
#     split_data = {
#         'train': train_set_original,
#         'test': test_set_original,
#     }

#     # Define the file path for saving the split data
#     data_file_path = 'dataset.pkl'

#     # Serialize and save the split data to the file
#     with open(data_file_path, 'wb') as file:
#         pickle.dump(split_data, file)

#     # Load the data into batches
#     train_set = torch.utils.data.DataLoader(train_set_original, batch_size=batch_size, shuffle=True)
#     test_set = torch.utils.data.DataLoader(test_set_original, batch_size=batch_size, shuffle=True)
#     # Print the layers in the model
#     model = VariationalAutoencoder(latent_dims)
#     for name, param in model.named_parameters():
#         print(name)
#     vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
#     vae = train(vae, train_set, test_set, epochs, 15, 5)





