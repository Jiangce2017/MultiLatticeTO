import pickle
import torch
import os.path as osp
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from VAE_Pytorch_Hybrid_to_Hybrid import VariationalAutoencoder
from utils import plot_latent
torch.manual_seed(1234)

def L2_dist(arr1,arr2):
    return torch.sqrt(torch.sum((arr1-arr2)**2,dim=-1))

def elasticity(x):
    # x (2D array): contains 1's and 0's
    # penal (positive float): affects penalization of cells, but has no effect here since no gray values

    penal = 1
    nely, nelx = np.shape(x)

    ## MATERIAL PROPERTIES
    EO = 1
    Emin = 1e-9
    nu = 0.3

    ## PREPARE FINITE ELEMENT ANALYSIS
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

    KE = 1 / (1 - nu**2) / 24 * (np.block([[A11, A12], [np.transpose(A12), A11]]) + nu * np.block([[B11, B12], [np.transpose(B12), B11]]))

    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((1 + nely, 1 + nelx), order="F")
    edofVec = np.reshape(2 * nodenrs[:-1, :-1] + 1, (nelx * nely, 1), order="F")
    edofMat = np.tile(edofVec, (1, 8)) + np.tile(np.concatenate(([0, 1], 2*nely+np.array([2,3,0,1]), [-2, -1])), (nelx*nely, 1))

    iK = np.reshape(np.kron(edofMat, np.ones((8, 1))).T, (64 * nelx * nely, 1), order='F') # Need order F to match the reshaping of Matlab
    jK = np.reshape(np.kron(edofMat, np.ones((1, 8))).T, (64 * nelx * nely, 1), order='F')

    ## PERIODIC BOUNDARY CONDITIONS
    e0 = np.eye(3)
    ufixed = np.zeros((8, 3))
    U = np.zeros((2*(nely+1)*(nelx+1), 3))

    alldofs = np.arange(1, 2*(nely+1)*(nelx+1)+1)
    n1 = np.concatenate((nodenrs[-1, [0, -1]], nodenrs[0, [-1,0]]))
    d1 = np.reshape(([[(2*n1-1)], [2*n1]]), (1, 8), order='F')  # four corner points of the object
    n3 = np.concatenate((nodenrs[1:-1, 0].T, nodenrs[-1, 1:-1]))
    d3 = np.reshape(([[(2*n3-1)], [2*n3]]), (1, 2*(nelx+nely-2)), order='F')  # Left and Bottom boundaries
    n4 = np.concatenate((nodenrs[1:-1, -1].flatten(), nodenrs[0, 1:-1].flatten()))
    d4 = np.reshape(([[(2*n4-1)], [2*n4]]), (1, 2*(nelx+nely-2)), order='F')  # Right and Top Boundaries
    d2 = np.setdiff1d(alldofs, np.hstack([d1, d3, d4]))  # All internal nodes in the shape

    for j in range(0, 3):
        ufixed[2:4, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([nelx, 0])
        ufixed[6:8, j] = np.array([[e0[0, j], e0[2, j]/2], [e0[2, j]/2, e0[1, j]]]) @ np.array([0, nely])
        ufixed[4:6, j] = ufixed[2:4, j]+ufixed[6:8, j]

    wfixed = np.concatenate((np.tile(ufixed[2:4, :], (nely-1, 1)), np.tile(ufixed[6:8, :], (nelx-1, 1))))

    ## INITIALIZE ITERATION
    qe = np.empty((3, 3), dtype=object)
    Q = np.zeros((3, 3))
    xPhys = x

    '''
    # For printing out large arrays
    with np.printoptions(threshold=np.inf):
        K_copy[np.abs(K_copy) < 0.000001] = 0
        print(K_copy)
    '''

    ## FE-ANALYSIS
    sK = (KE.flatten(order='F')[:, np.newaxis] * (Emin+xPhys.flatten(order='F').T**penal*(EO-Emin)))[np.newaxis, :].reshape(-1, 1, order='F')

    K = sp.coo_matrix((sK.flatten(order='F'), (iK.flatten(order='F')-1, jK.flatten(order='F')-1)), shape=(np.shape(alldofs)[0], np.shape(alldofs)[0]))
    K = (K + K.transpose())/2
    K = sp.csr_matrix(np.nan_to_num(K))  # Remove the NAN values and replace them with 0's and large finite values
    K = K.tocsc()  # Converting to csc from csr transposes the matrix to match the orientation in MATLAB


    k1 = K[d2-1][:, d2-1]
    k2 = (K[d2[:,None]-1, d3-1] + K[d2[:,None]-1, d4-1])
    k3 = (K[d3-1, d2[:, None]-1] + K[d4-1, d2[:, None]-1])
    k4 = K[d3-1,d3.T-1] + K[d4-1, d4.T-1] + K[d3-1, d4.T-1] + K[d4-1, d3.T-1]

    Kr_top = sp.hstack((k1, k2))
    Kr_bottom = sp.hstack((k3.T, k4))
    Kr = sp.vstack((Kr_top, Kr_bottom)).tocsc()

    U[d1-1, :] = ufixed
    U[np.concatenate((d2-1, d3.ravel()-1)), :] = spsolve(Kr, ((-sp.vstack((K[d2[:, None]-1, d1-1], (K[d3-1, d1.T-1]+K[d4-1, d1.T-1]).T)))*ufixed-sp.vstack((K[d2-1, d4.T-1].T, (K[d3-1, d4.T-1]+K[d4-1, d4.T-1]).T))*wfixed))
    U[d4-1, :] = U[d3-1, :]+wfixed


    ## OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    for i in range(0, 3):
        for j in range(0, 3):
            U1 = U[:, i] # not quite right
            U2 = U[:, j]
            qe[i, j] = np.reshape(np.sum((U1[edofMat-1] @ KE) * U2[edofMat-1], axis=1), (nely, nelx), order='F') / (nelx * nely)
            Q[i, j] = sum(sum((Emin+xPhys**penal*(EO-Emin))*qe[i, j]))
    Q[np.abs(Q) < 0.000001] = 0

    return Q


class MaterialModel:
    def __init__(self,config,device):
        self.searchMode = config.searchMode
        self.simplexDim = config.simplexDim
        self.latentrDim = config.latentDim
        # Load the split data from the file
        #data_file_path = "dataset_strut.pkl"
        with open(config.lattice_data_file_path, 'rb') as file:
            dataset = pickle.load(file)

        # Define the labels in the saved dataset
        train = dataset["train"]  # contains indices: 0 = unit cells, 1 = stiffness tensors, 2 = volume fractions
        test = dataset["test"] 
        print("total train data:{}".format(len(train)) )
        print(train[100][0])
        print(train[0][0].shape)
        latent_dimensionality = 16
        vae  = VariationalAutoencoder(latent_dimensionality).to(device)

        # # Load the model from a checkpoint
        # checkpoint = torch.load(lattice_model_file)
        # # Load the stiffness normalizer from the checkpoint
        # scaler = checkpoint['scaler']
        # # Load the model parameters from the checkpoint
        # model_parameters = checkpoint['model']
        # vae.load_state_dict(model_parameters)

        with open(config.vae_file_path, "rb") as fp:
            model_parameters = pickle.load(fp)
            vae.load_state_dict(model_parameters['model'])
            scaler = model_parameters['scaler']
        
        self.vae = vae
        self.scaler = scaler
        latent_points_original = plot_latent(vae, train, scaler)


        latent_points_arr = torch.cat(latent_points_original,dim=0)

        print("total sample points: {}".format(latent_points_arr.shape[0]))
        print(latent_points_arr.shape)
        print(latent_points_arr[:3,:])
        dist_min = torch.min(L2_dist(latent_points_arr[1:,:],latent_points_arr[:-1,:]))
        dist_max = torch.max(L2_dist(latent_points_arr[1:,:],latent_points_arr[:-1,:]))
        for i_point in range(latent_points_arr.shape[0]-1):
            for j_point in range(i_point+1, latent_points_arr.shape[0]):
                dist = L2_dist(latent_points_arr[i_point],latent_points_arr[j_point])
                if dist < dist_min:
                    dist_min = dist
                if dist > dist_max:
                    dist_max = dist
        
        print("dist_min:{}".format(dist_min))
        print("dist_max:{}".format(dist_max))
        
        if self.searchMode == 'simplex':
            # # Select random data points for interpolating 
            # idx = np.random.permutation(range(latent_points_arr.shape[0]))
            # latent_points_arr = latent_points_arr[idx]
            # simplex_points = latent_points_arr[:self.simplexDim+1,:]

            # radius = 2*dist_min
            # neighbors = torch.zeros(latent_points_arr.shape[0])
            # for i_point in range(latent_points_arr.shape[0]):
            #     dist2points = L2_dist(latent_points_arr,latent_points_arr[[i_point],:])
            #     neighbors[i_point]=torch.sum(dist2points < radius)
            # # find the points have the most points in its neighbor ball
            # print("the most number of neighbors: {}".format(torch.max(neighbors,dim=0).values))
            # nearest_center_point = torch.max(neighbors,dim=0).indices       
            
            radius = 3*dist_min
            virtual_center = torch.mean(latent_points_arr,dim=0)
            dist_2_virtual_center = L2_dist(latent_points_arr,virtual_center)
            nearest_center_point = torch.max(dist_2_virtual_center,dim=0).indices
            center = latent_points_arr[nearest_center_point,:]
            simplex_points = torch.zeros((self.simplexDim+1,latent_points_arr.shape[1]))
            dist_2_center = L2_dist(latent_points_arr,center)
            dist_2_center[dist_2_center>radius] = 0
            first_simplex_point = torch.max(dist_2_center,dim=0).indices
            simplex_points[0,:] = latent_points_arr[first_simplex_point,:]
            candidates = latent_points_arr[dist_2_center<radius]
            for i_dim in range(1,self.simplexDim+1):
                L = 0
                for i_point in range(i_dim):
                    L += L2_dist(candidates,simplex_points[i_point,:])
                farest_point = torch.max(L,dim=0).indices
                simplex_points[i_dim,:] = candidates[farest_point,:]



            # neighbors = torch.zeros(latent_points_arr.shape[0])
            # for i_point in range(latent_points_arr.shape[0]):
            #     dist2points = L2_dist(latent_points_arr,latent_points_arr[[i_point],:])
            #     neighbors[i_point]=torch.sum(dist2points < radius)
            # # find the points have the most points in its neighbor ball
            # print("the most number of neighbors: {}".format(torch.max(neighbors,dim=0).values))
            # point = torch.max(neighbors,dim=0).indices
            # #### select self.simplexDim + 1 points
            # simplex_points = torch.zeros((self.simplexDim+1,latent_points_arr.shape[1]))
            # simplex_points[0,:] = latent_points_arr[point,:]
            # select_mask = torch.arange(latent_points_arr.shape[0]) != point
            # latent_points_arr = latent_points_arr[select_mask,:]
            # for i_dim in range(1,self.simplexDim+1):
            #     dist = L2_dist(latent_points_arr,simplex_points[[i_dim-1],:])
            #     nearest_point = torch.max(dist,dim=0).indices
            #     simplex_points[i_dim,:] = latent_points_arr[nearest_point]
            #     select_mask = torch.arange(latent_points_arr.shape[0]) != nearest_point
            #     latent_points_arr = latent_points_arr[select_mask,:]

            
            interpolate_list, nn_C, v = self.vae.decoder(simplex_points, self.scaler)

            self.simplex_points = simplex_points
            #print("plot lattices")
            
            fig, ax = plt.subplots(1,self.simplexDim+1)
            cmap = 'Oranges'
            cmap = plt.get_cmap(cmap) 
            for i_point in range(self.simplexDim+1):
                ax[i_point].imshow(interpolate_list[i_point].view(28, 28).cpu().detach().numpy(),cmap=cmap, vmin=0, vmax=1.0)
                ax[i_point].axis("off")
            fig.savefig(osp.join(config.results_dir, 'lattices_'+str(config.simplexDim+1)),dpi = 450)
        elif  self.searchMode == 'cubic':
            #### select a point that have the most neighbor
            print("total sample points: {}".format(latent_points_arr.shape[0]))
            dist_min = torch.min(L2_dist(latent_points_arr[1:,:],latent_points_arr[:-1,:]))
            dist_max = torch.max(L2_dist(latent_points_arr[1:,:],latent_points_arr[:-1,:]))
            print("dist_min:{}".format(dist_min))
            print("dist_max:{}".format(dist_max))
            radius = 1.3*dist_min
            neighbors = torch.zeros(latent_points_arr.shape[0])
            for i_point in range(latent_points_arr.shape[0]):
                dist2points = L2_dist(latent_points_arr,latent_points_arr[[i_point],:])
                neighbors[i_point]=torch.sum(dist2points < radius)
            # find the points have the most points in its neighbor ball
            print("the most number of neighbors: {}".format(torch.max(neighbors,dim=0).values))
            point = torch.max(neighbors,dim=0).indices
            origin = latent_points_arr[point,:]
            self.radius = radius
            self.origin = origin     
            #### build the spherical coordinate
        

    def map2material(self,nn_t):
        if self.searchMode == 'simplex':
            latent_vec = torch.einsum('ij,jk->ik',nn_t,self.simplex_points)
        elif self.searchMode == 'cubic':
            latent_vec = self.origin + self.radius*(2*nn_t-1)
        # if self.latentParDim == 1:
        #     latent_vec = self.z_1 + (self.z_2 - self.z_1)*nn_t     
        # elif  self.latentParDim == 16:
        #     latent_vec = self.origin + self.radius*(2*nn_t-1)
        interpolate_list, nn_C, v = self.vae.decoder(latent_vec, self.scaler)
        return interpolate_list, nn_C, v 
    
    def checkStiffness(self,nn_t):
        interpolate_list, nn_C, v = self.map2material(nn_t)
        interpolate_list = interpolate_list.detach().cpu().numpy()
        nn_C = nn_C.detach().cpu().numpy()
        true_C = np.zeros(nn_C.shape)
        for i_lattice in range(interpolate_list.shape[0]):
            Q = elasticity(interpolate_list[i_lattice])
            true_C[i_lattice,:,:] = Q
            ### compare Q and nn_C[i_lattice]


            