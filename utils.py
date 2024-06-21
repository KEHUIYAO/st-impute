import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class STDataset(Dataset):
    def __init__(self, X, y):
        # Initialize the dataset with input features X and target variable y
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))
  
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]



# lightning data module
class STDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super(STDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)




def interpolate_missing_values(y_st, y_st_missing, mask_st):
    num_nodes, seq_len = y_st.shape
    for k in range(num_nodes):
        for l in range(seq_len):
            y_st_missing[k, :] = pd.Series(y_st_missing[k, :]).interpolate(method='linear', limit_direction='both').values
    y_st_missing[np.isnan(y_st_missing)] = np.nanmean(y_st_missing)

    return y_st_missing

def generate_space_basis_functions(space_coords):
    num_nodes = space_coords.shape[0]

    # spatial basis functions
    num_basis = [10**2,19**2,37**2]
    knots_1d = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
    # Wendland kernel
    K = 0
    space_basis = np.zeros((num_nodes, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/np.sqrt(num_basis[res])*2.5
        knots_s1, knots_s2 = np.meshgrid(knots_1d[res],knots_1d[res])
        knots = np.column_stack((knots_s1.flatten(),knots_s2.flatten()))
        for i in range(num_basis[res]):
            d = np.linalg.norm(space_coords-knots[i,:],axis=1)/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    space_basis[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    space_basis[j,i + K] = 0
        K = K + num_basis[res]
    return space_basis

def generate_time_basis_functions(time_coords):
    seq_len = time_coords.shape[0]
    # time basis functions
    time_coords = time_coords.reshape(-1, 1)

    num_basis = [10,19,37,73]
    knots = [np.linspace(0,1,i) for i in num_basis]
    # Wendland kernel
    K = 0 # basis size
    time_basis = np.zeros((seq_len, sum(num_basis)))
    for res in range(len(num_basis)):
        theta = 1/num_basis[res]*2.5
        for i in range(num_basis[res]):
            d = np.absolute(time_coords-knots[res][i])/theta
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    time_basis[j,i + K] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
                else:
                    time_basis[j,i + K] = 0
        K = K + num_basis[res]
    return time_basis


def convert_st_data_for_dnn_training(y_st, mask_st, space_coords, time_coords, covariate='coord'):
    num_nodes, seq_len = y_st.shape
    y = y_st.reshape(-1)
    mask = mask_st.reshape(-1)

    # scale time coordinates to [0, 1]
    time_coords = (time_coords - time_coords.min()) / (time_coords.max() - time_coords.min())

    # scale space coordinates to [0, 1]
    space_coords = (space_coords - space_coords.min(axis=0)) / (space_coords.max(axis=0) - space_coords.min(axis=0))

    if covariate == 'coord':
        spatial_covariate = space_coords
        time_covariate = time_coords.reshape(-1, 1)

    elif covariate == 'time_basis':
        time_covariate = generate_time_basis_functions(time_coords)
        spatial_covariate = space_coords

    elif covariate == 'space_basis':
        spatial_covariate = generate_space_basis_functions(space_coords)
        time_covariate = time_coords.reshape(-1, 1)

    
    elif covariate == 'st_basis':
        time_covariate = generate_time_basis_functions(time_coords)
        spatial_covariate = generate_space_basis_functions(space_coords)
        

    spatial_covariate_expand = []
    time_covariate_expand = []
    for i in range(num_nodes):
        for j in range(seq_len):
            spatial_covariate_expand.append(spatial_covariate[i, :])
            time_covariate_expand.append(time_covariate[j, :])

    # convert spatial and time covariates to numpy arrays
    spatial_covariate_expand = np.stack(spatial_covariate_expand, axis=0)
    time_covariate_expand = np.stack(time_covariate_expand, axis=0)

    X = np.concatenate([spatial_covariate_expand, time_covariate_expand], axis=1)
    
    # split X and y into training and testing sets based on mask
    X_train = X[mask == 1, :]
    y_train = y[mask == 1]
    X_test = X[mask == 0, :]
    y_test = y[mask == 0]

    return X_train, y_train, X_test, y_test