from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt


########################################################################################################################
# Read an array from the Dataframe converted from a CSV
def read_df_csv(dataframe: pd.DataFrame, row: int, column: int or str) -> np.ndarray:
    """
    :param dataframe: [pd.Dataframe] - a Pandas Dataframe consisting of JSON encoded arrays
    :param row: [int] - the data point desired to convert in the Dataframe
    :param column: [int or str] - the column index or name that the datapoint is contained in
    :return: [ndarray] - the function will return the numpy version of the array by reading the JSON string
    """
    json_array = dataframe.iloc[row][column]
    array = np.array(json.loads(json_array))
    return array


########################################################################################################################
# Define the Dataset Class to be used by the model
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load the data from a location
        self.df = pd.read_csv(csv_file)  # read the csv file
        # self.root_dir = root_dir  # supply a directory to reference
        self.transform = transform  # use to define a function to transform the data

        # Designate the dataframes
        self.shape_array = pd.DataFrame(self.df.iloc[:, 0]) # in the CSV file, the first column contains the shape data
        self.stiffness_array = pd.DataFrame(self.df.iloc[:, 1])  # in the CSV file, the second column contains the property data
        # Initialize an empty list to store stiffness values
        self.all_stiffness_values = []
        # can define further properties in the future

    def __len__(self):
        return len(self.shape_array)

    def __getitem__(self, idx):
        shape = read_df_csv(self.shape_array, idx, 0)
        stiffness = read_df_csv(self.stiffness_array, idx, 0)
        if shape.dtype != np.float32:
            shape = shape.astype(np.float32)
            stiffness = stiffness.astype(np.float32)

        # Append the stiffness values to the list
        self.all_stiffness_values.append(stiffness.flatten().tolist())

        # Determine the Volume Fraction of the data
        volume_fraction = np.array(shape > 0, dtype=int)  # Convert all values in array to 1
        volume_fraction = np.mean(volume_fraction, axis=(0, 1))  # Average the values in each unit cell to "count"
        volume_fraction = volume_fraction.astype(np.float32)

        # if self.transform is not None:  # We may need to apply the tensor transform like MNIST??
        #     shape = self.transform(shape)

        # sample = {'shape': shape, 'stiffness': stiffness, 'volume_fraction':volume_fraction}
        # return sample
        return shape, stiffness, volume_fraction

    def get_all_stiffness_values(self):
        return self.all_stiffness_values
########################################################################################################################
# EX: How to use the dataset class
'''
dataset = CustomDataset("C:/Users/balma/Pycharm_Stuff/PycharmProjects/AutoEncoders_2D_Stiffness_Old_Data/Old_VAE_Dataset/Combined_Space.csv")

test_point = 0
sample = dataset[test_point]  # Index a desired sample
test_shape = sample['shape']  # Assign the specific sample shape
test_stiffness = sample['stiffness']  # Assign the specific sample properties

# View the characteristics of the samples
print('Data Type:', type(test_shape))
print('Shape of Unit Cells:', np.shape(test_shape))
print('Shape of Unit Cell Elasticity Tensor:', np.shape(test_stiffness))

# Display the output shape
plt.imshow(test_shape)
plt.show()

print('Elasticity Test Sample\n', test_stiffness)
'''