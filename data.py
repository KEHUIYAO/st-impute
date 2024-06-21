import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky



class GP():
    def __init__(self, num_nodes, seq_len):
        self.num_nodes = num_nodes
        self.seq_len = seq_len

       
    def matern_covariance(self, x1, x2, length_scale=1.0, nu=0.5, sigma=1.0):
        dist = np.linalg.norm(x1 - x2)
        if dist == 0:
            return sigma ** 2
        coeff1 = (2 ** (1 - nu)) / gamma(nu)
        coeff2 = (np.sqrt(2 * nu) * dist) / length_scale
        return sigma ** 2 * coeff1 * (coeff2 ** nu) * kv(nu, coeff2)


    def generate_st_data(self, s_l=1, s_nu=0.5, t_l=5, t_nu=0.5, s_sigma=1, t_sigma=1, seed=42):

        seq_len = self.seq_len
        num_nodes = self.num_nodes

        rng = np.random.RandomState(seed)

        time_coords = np.arange(0, seq_len)
        space_coords = np.random.rand(num_nodes, 2)
        

        # create the temporal covariance matrix
        var_temporal = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                var_temporal[i, j] = self.matern_covariance(time_coords[i], time_coords[j], t_l, t_nu, t_sigma)

        # create the spatial covariance matrix
        var_spatial = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                var_spatial[i, j] = self.matern_covariance(space_coords[i, :], space_coords[j, :], s_l, s_nu, s_sigma)

        L_spatial = cholesky(var_spatial + 1e-6 * np.eye(num_nodes), lower=True)
        L_temporal = cholesky(var_temporal + 1e-6 * np.eye(seq_len), lower=True)

        eta = rng.normal(0, 1, num_nodes * seq_len)
        L_spatial_temporal = np.kron(L_spatial, L_temporal)
        y = np.einsum('ij, j->i', L_spatial_temporal, eta)
        y_st = y.reshape(num_nodes, seq_len)

        return y_st, space_coords, time_coords
    
    def generate_st_data_with_missing_values(self, s_l=5, s_nu=0.5, t_l=5, t_nu=0.5, s_sigma=1, t_sigma=1, missing_rate=0.1, seed=42):
        rng = np.random.RandomState(seed)
        y_st, space_coords, time_coords = self.generate_st_data(s_l, s_nu, t_l, t_nu, s_sigma, t_sigma, seed)
        mask_st = rng.choice([0, 1], size=(self.num_nodes, self.seq_len), p=[missing_rate, 1 - missing_rate])
        y_st_missing = y_st.copy()
        y_st_missing[mask_st == 0] = np.nan
        return y_st, y_st_missing, mask_st, space_coords, time_coords

    

if __name__ == "__main__":
    num_nodes = 3
    seq_len = 4
    gp = GP(num_nodes, seq_len)
    
    
    # generate data with strong temporal correlation and weak spatial correlation
    y_st, y_st_missing, mask_st, space_coords, time_coords = gp.generate_st_data_with_missing_values(t_l=10, s_l=0.1)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.imshow(y_st_missing, aspect='auto')
    plt.colorbar()
    plt.title('ST Data with Missing Values')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

     # generate data with weak temporal correlation and strong spatial correlation
    y_st, y_st_missing, mask_st, space_coords, time_coords = gp.generate_st_data_with_missing_values(t_l=1, s_l=1.2)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.imshow(y_st_missing, aspect='auto')
    plt.colorbar()
    plt.title('ST Data with Missing Values')
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.show()

    