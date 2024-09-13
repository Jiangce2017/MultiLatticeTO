import torch
import numpy as np
import random
torch.manual_seed(1234)

def plot_latent(autoencoder, data, scaler, num_batches=100,device=torch.device("cpu")):
    latent_points = []
    data = torch.utils.data.DataLoader(data, shuffle=True)
    for i, (x, y, volume_fraction) in enumerate(data):
        z = autoencoder.encoder(x.to(device), y.to(device), volume_fraction.to(device), scaler)
        z = z.to('cpu').detach()
        latent_points.append(z)
        #plt.scatter(z[:, 0], z[:, 1])#, c=y, cmap='tab10')
        if i > 10*num_batches:
            #plt.colorbar()
            break
    return latent_points

#%%  set device CPU/GPU
def setDevice(overrideGPU = True):
    if(torch.cuda.is_available() and (overrideGPU == False) ):
        device = torch.device("cuda:0")
        print("GPU enabled")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device

#%% Seeding
def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)

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

