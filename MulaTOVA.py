import numpy as np
import random
import torch
import time
import torch.nn as nn
import torch.optim as optim
from os import path
from FE import StructuralFE
import matplotlib.pyplot as plt
from matplotlib import colors
from pytictoc import TicToc
import os.path as osp
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from TO_models import TopNet
from utils import PytorchMinMaxScaler, plot_latent,setDevice,set_seed
from material_models import MaterialModel
from material_models import elasticity

from matplotlib import rc
rc('text', usetex=False)
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 150
timer = TicToc()

overrideGPU = False
device = setDevice(overrideGPU) 
torch.autograd.set_detect_anomaly(True)
    
class TopologyOptimizer:
    def __init__(self,config):
        self.nelx = config.nelx
        self.nely = config.nely
        self.cell_width = config.cell_width
        self.cell_type = config.cell_type
        self.results_dir = config.results_dir
        self.interactive = config.interactive
        self.desiredVolumeFraction = config.desiredVolumeFraction
        self.exper_name = self.exampleName + "_" + config.nn_type+ "_"+config.cell_type + "_" + str(config.desiredVolumeFraction) 
        self.selecting_loading(config.example)
        self.initializeFE(config)
        self.initializeOptimizer(config)
        self.InitializeMaterialModel(config,device)
        

    def initializeFE(self,config):
        self.FE = StructuralFE() 
        self.FE.initializeSolver(config.nelx, config.nely, self.force, self.fixed, config.penal, config.Emin, config.Emax) 
        self.xy, self.nonDesignIdx = self.generatePoints(config.nelx, config.nely, 1, self.nonDesignRegion) 
        self.xyPlot, self.nonDesignPlotIdx = self.generatePoints(config.nelx, config.nely, config.cell_width, self.nonDesignRegion) 
    
    def initializeOptimizer(self, config):
        self.density = config.desiredVolumeFraction*np.ones((self.nelx*self.nely))
        self.topNet = TopNet(config,self.symXAxis,self.symYAxis).to(device)
        self.objective = 0.
        self.convergenceHistory = []

    def InitializeMaterialModel(self,config,device):
        self.material_model = MaterialModel(config,device)

    def optimizeDesign(self,config):
        self.convergenceHistory = [] 
        savedNetFileName = osp.join(config.results_dir, self.exampleName + '_'  + str(self.nelx) + '_' + str(self.nely) +  '.nt')
        savedMaterialNetFileName = osp.join(config.results_dir,self.exampleName + '_'  + str(self.nelx) + '_' + str(self.nely) +  'material.nt' )
        alphaMax = 100*config.desiredVolumeFraction 
        alphaIncrement= 0.08
        alpha = alphaIncrement  # start
        nrmThreshold = 0.1  # for gradient clipping
        if(config.useSavedNet):
            if (path.exists(savedNetFileName)):
                self.topNet = torch.load(savedNetFileName) 
                self.material_model = torch.load(savedMaterialNetFileName)
            else:
                print("Network file not found") 
        self.optimizer = optim.Adam(self.topNet.parameters(),lr=config.learningRate)
        w = self.cell_width
        batch_x =  self.xy.view(-1,2).float().to(device)
        for epoch in range(config.maxEpochs):
            self.optimizer.zero_grad()
            nn_rho,nn_t = self.topNet(batch_x,1,self.nonDesignIdx)
            if config.cell_type == "lattice": 
                interpolate_list, nn_C, v = self.material_model.map2material(nn_t)
                true_v = torch.sum(torch.sum(interpolate_list,dim=2),dim=1)/(w*w)
                true_rho = nn_rho*true_v 
                u,Jelem = self.FE.solvelatticetorch(nn_rho,nn_C)
                compliance = torch.sum(self.FE.Emax*(nn_rho**self.FE.penal)*Jelem)
            else:
                u,Jelem = self.FE.solvetorch(nn_rho)
                true_rho = nn_rho
                compliance = torch.sum(self.FE.Emax*(nn_rho**self.FE.penal)*Jelem)  # compliance
            if(epoch == 0):
                self.obj0 = torch.sum(self.FE.Emax*(nn_rho**self.FE.penal)*Jelem).detach()
            objective = compliance/self.obj0
            
            volConstraint =((torch.mean(true_rho)/config.desiredVolumeFraction) - 1.0) 
            currentVolumeFraction = torch.mean(true_rho).item() 
            self.objective = objective
            loss = self.objective+ alpha*pow(volConstraint,2)
            alpha = min(alphaMax, alpha + alphaIncrement) 
            loss.backward(retain_graph=True) 
            torch.nn.utils.clip_grad_norm_(self.topNet.parameters(),nrmThreshold)
            self.optimizer.step()
            if(volConstraint < 0.05): # Only check for gray when close to solving. Saves computational cost  
                greyElements = torch.sum((nn_rho > 0.05)*(nn_rho < 0.95)).item()  
                relGreyElements = greyElements/nn_rho.shape[0]
            else:
                relGreyElements = 1 
            self.convergenceHistory.append([ self.objective.item(), currentVolumeFraction,loss.item(),relGreyElements]) 
            self.FE.penal = min(4.0,self.FE.penal + 0.01)  # continuation scheme
            if(epoch % 10 == 0):
                if config.interactive:
                    self.plotTO(epoch) 
                print("{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} "\
                  .format(epoch, self.objective.item()*self.obj0 ,currentVolumeFraction,loss.item(),relGreyElements))
            if ((epoch > config.minEpochs ) & (relGreyElements < 0.035) & (volConstraint< 0) ):
                break 
        self.plotTO(epoch,True) 
        print("{:3d} J: {:.2F}; Vf: {:.3F}; loss: {:.3F}; relGreyElems: {:.3F} "\
             .format(epoch, self.objective.item()*self.obj0 ,currentVolumeFraction,loss.item(),relGreyElements))  
        torch.save(self.topNet, savedNetFileName)
        torch.save(self.material_model, savedMaterialNetFileName)
        ### save data

    def plotTO(self, iter,saveFig= False):
        saveFrame = True  # set this T/F if you want to create frames- use for video
        w = self.cell_width
        batch_x = self.xy.view(-1,2).float().to(device)  
        nn_rho,nn_t = self.topNet(batch_x,1,self.nonDesignIdx)
        nn_rho = nn_rho.to('cpu').detach().numpy()
        if self.cell_type == "lattice":     
            interpolate_list, nn_C, v = self.material_model.map2material(nn_t)
            interpolate_list_np = interpolate_list.detach().numpy()
            true_v = np.sum(np.sum(interpolate_list_np,axis=2),axis=1)/(w*w) 
            true_rho = nn_rho*true_v 
            img = np.zeros(((self.FE.nely)*w, (self.FE.nelx)*w))
            for i, x_hat in enumerate(interpolate_list_np):
                block_x = self.FE.nely - i % self.FE.nely
                block_y = i // self.FE.nely
                img[(block_x-1)*w:block_x*w,block_y*w:(block_y+1)*w] = np.flip(x_hat.transpose(),axis=0)*nn_rho[i]
        else:
            img = np.flip(nn_rho.reshape(self.FE.nelx,self.FE.nely).transpose(),axis=0)
            true_rho = nn_rho
        if self.interactive:
            plt.ion() 
        plt.clf()
        if (saveFig):
            example = self.example
            if self.cell_type == "lattice":
                real_compliance,dist = self.full_structure_FE(example,self.cell_type,nn_rho,w,interpolate_list)
            else:
                real_compliance,dist = self.full_structure_FE(example,self.cell_type,nn_rho,w)
            print("real compliance is: {}".format(real_compliance))    
        else:
             real_compliance = self.objective*self.obj0
        plt.xticks([])
        plt.yticks([])
        plt.title('Iter = {:d}, J = {:.2F}, V_f = {:.2F}, V_des = {:.2F}'.format(iter, real_compliance, np.mean(true_rho),  self.desiredVolumeFraction),loc='left')
        plt.grid(False)
        axes = plt.gca()
        cmap = 'Oranges'
        cmap = plt.get_cmap(cmap) 
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cbar = plt.colorbar(m, cax=cax, aspect=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Density", fontsize=10)
        plt.ticklabel_format(style="plain")
        axes.imshow(img,cmap=cmap,vmin=0,vmax=1)      
        if(saveFrame):
            frame_file_name = osp.join(self.results_dir, 'frames','f_'+'+str(iter)'+'.jpg')
            plt.savefig(frame_file_name)            
        if (saveFig):    
            fName = osp.join(self.results_dir, self.exampleName+'_topology.png')
            plt.savefig(fName,dpi = 450)
            data_file_nme = osp.join(self.results_dir, self.exampleName+'_img.npy')
            np.save(data_file_nme,img,allow_pickle=False)
        if self.interactive:  
            plt.pause(0.01)

    def plotTO_smooth(self, iter,saveFig= False):
        saveFrame = False  # set this T/F if you want to create frames- use for video
        w = self.cell_width
        batch_x = self.xy.view(-1,2).float().to(device)  
        nn_rho,nn_t = self.topNet(batch_x,1,self.nonDesignIdx)
        nn_rho = nn_rho.to('cpu').detach().numpy()
        if self.cell_type == "lattice":     
            interpolate_list, nn_C, v = self.material_model.map2material(nn_t)
            interpolate_list_np = interpolate_list.detach().numpy()
            true_v = np.sum(np.sum(interpolate_list_np,axis=2),axis=1)/(w*w) 
            true_rho = nn_rho*true_v 
            lattice_img = np.zeros(((self.FE.nely)*w, (self.FE.nelx)*w))
            solid_img = np.zeros(((self.FE.nely)*w, (self.FE.nelx)*w))
            for i, x_hat in enumerate(interpolate_list_np):
                block_x = self.FE.nely - i % self.FE.nely
                block_y = i // self.FE.nely
                lattice_img[(block_x-1)*w:block_x*w,block_y*w:(block_y+1)*w] = np.flip(x_hat.transpose(),axis=0)*nn_rho[i]
                solid_img[(block_x-1)*w:block_x*w,block_y*w:(block_y+1)*w] = np.ones(x_hat.shape)*nn_rho[i]
            img = lattice_img
        else:
            solid_img = np.flip(nn_rho.reshape(self.FE.nelx,self.FE.nely).transpose(),axis=0)
            true_rho = nn_rho
            img = solid_img
        large_batch_x = self.xyPlot.view(-1,2).float().to(device)
        large_nn_rho,large_nn_t = self.topNet(large_batch_x,w,self.nonDesignPlotIdx)
        #nn_rho,nn_t = self.topNet(batch_x,1,self.nonDesignIdx)
        large_nn_rho = large_nn_rho.to('cpu').detach().numpy()
        large_img = np.flip(large_nn_rho.reshape(self.FE.nelx*w,self.FE.nely*w).transpose(),axis=0)
        large_img[large_img>0.1] = 1
        large_img[large_img<0.1] = 0
        if self.cell_type == "lattice":
            solid_img[solid_img>0.5] = 1
            solid_img[solid_img<0.5] = 0
            lattice_img[lattice_img>0.5] = 1
            lattice_img[lattice_img<0.5] = 0
            large_img = large_img-solid_img+lattice_img
        img = large_img

        if self.interactive:
            plt.ion() 
        
        plt.clf()
        if (saveFig):
            example = self.example
            if self.cell_type == "lattice":
                real_compliance,dist = self.full_structure_FE(example,self.cell_type,nn_rho,w,interpolate_list)
            else:
                real_compliance,dist = self.full_structure_FE(example,self.cell_type,nn_rho,w)
            print("real compliance is: {}".format(real_compliance))    
        else:
             real_compliance = self.objective*self.obj0
        plt.xticks([])
        plt.yticks([])
        plt.title('Iter = {:d}, J = {:.2F}, V_f = {:.2F}, V_des = {:.2F}'.format(iter, real_compliance, np.mean(true_rho),  self.desiredVolumeFraction),loc='left')
        plt.grid(False)
        axes = plt.gca()
        cmap = 'Oranges'
        cmap = plt.get_cmap(cmap) 
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cbar = plt.colorbar(m, cax=cax, aspect=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Density", fontsize=10)
        plt.ticklabel_format(style="plain")
        axes.imshow(img,cmap=cmap,vmin=0,vmax=1)  
        if(saveFrame):
            frame_file_name = osp.join(self.results_dir, 'frames','f_'+'+str(iter)'+'.jpg')
            plt.savefig(frame_file_name)             
        if (saveFig):    
            fName = osp.join(self.results_dir, self.exampleName+'_topology.png')
            plt.savefig(fName,dpi = 450)
            data_file_nme = osp.join(self.results_dir, self.exampleName+'_img.npy')
            np.save(data_file_nme,img,allow_pickle=False)
        if self.interactive:  
            plt.pause(0.01)

    def plotConvergence(self):
        self.convergenceHistory = np.array(self.convergenceHistory) 
        plt.figure()
        plt.semilogy(self.convergenceHistory[:,0], 'b:',label = 'Rel. Compliance')
        plt.semilogy(self.convergenceHistory[:,1], 'r--',label = 'Vol. Fraction')
        plt.title('Convergence Plots' ) 
        plt.title('Convergence plots; V_des = {:.2F}'.format(self.desiredVolumeFraction))
        plt.xlabel('Iterations') 
        plt.grid('True')
        plt.legend(loc='lower left', shadow=True, fontsize='large')
        fName = osp.join(self.results_dir,self.exper_name+'_convergence.png')
        plt.savefig(fName,dpi = 450)

    def setup_structuralFE(self,example,nelx,nely,penal):
        large_ndof = 2*(nelx+1)*(nely+1) 
        large_force = np.zeros((large_ndof,1))
        large_dofs=np.arange(large_ndof)
        if example == 1:
            large_fixed = large_dofs[0:2*(nely+1):1] 
            large_force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1
            loading_point = 2*(nelx+1)*(nely+1)-2*nely+1
        if example == 2:
            large_fixed = large_dofs[0:2*(nely+1):1] 
            large_force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1 
            loading_point = 2*(nelx+1)*(nely+1)- (nely+1)
        if example == 3:
            large_fixed = np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1) 
            large_force[2*(nely+1)+1 ,0]=-1
            loading_point = 2*(nely+1)+1
        if example == 4:
            large_fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] ) 
            large_force[nelx*(nely+1)+1 ,0]=-1 
            loading_point = nelx*(nely+1)+1
        if example == 5:
            large_fixed = np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] ) 
            large_force[2*nely+1:2*(nelx+1)*(nely+1):2*(nely+1),0]=-1/(nelx+1)
            loading_point = 2*(nelx+1)*(nely+1)-1
        if example == 6:
            large_fixed =np.union1d(np.arange(0,2*(nely+1),2), 1)  # fix X dof on left
            large_force[2*(nelx+1)*(nely+1)- (nely), 0 ] = 1 
            loading_point = 2*(nelx+1)*(nely+1)- (nely)
        if example == 7:
            large_fixed = large_dofs[0:2*(nely+1):1] 
            large_force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1
            large_force[2*(nelx+1)*(nely+1)-2, 0 ] = 1 
            loading_point = 2*(nelx+1)*(nely+1)-2*nely+1
        if example == 8: 
            large_fixed = large_dofs[0:2*(nely+1):1] 
            large_force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1
            #large_force[2*(nelx+1)*(nely+1)- (nely), 0 ] = 1
            large_force[2*(nelx+1)*(nely+1)-2, 0 ] = 1 
            loading_point = 2*(nelx+1)*(nely+1)- (nely+1)
        if example == 9:
            large_fixed = np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1) 
            large_force[2*(nely+1)+1 ,0]=-1
            large_force[2*(nelx+1)*(nely+1)-2, 0 ] = 1 
            loading_point = 2*(nely+1)+1

        FE_solver = StructuralFE() 
        FE_solver.initializeSolver(nelx, nely, large_force, large_fixed, penal , Emin = 1e-6, Emax = 1.0)
        FE_solver.loading_point = loading_point
        return FE_solver
    
    def full_structure_FE(self,example,cell_type,nn_rho,w,interpolate_list=None):
        img = torch.zeros(((self.FE.nely)*w, (self.FE.nelx)*w))
        for i in range(nn_rho.shape[0]):
            block_x = self.FE.nely - i % self.FE.nely
            block_y = i // self.FE.nely
            if cell_type == "lattice":
                x_hat = interpolate_list[i]
                img[(block_x-1)*w:block_x*w,block_y*w:(block_y+1)*w] = torch.flip(torch.transpose(x_hat,0,1),dims=(0,))*nn_rho[i]
            else:
                img[(block_x-1)*w:block_x*w,block_y*w:(block_y+1)*w] = torch.ones((w,w))*nn_rho[i]
        large_rho = torch.transpose(torch.flip(img,dims=(0,)),0,1).reshape((self.FE.nelx*self.FE.nely*w*w)).cpu().detach().numpy()

        large_FE = self.setup_structuralFE(example,self.FE.nelx*w,self.FE.nely*w,self.FE.penal)
        large_u,large_Jelem = large_FE.solve88(large_rho)
        compliance = np.sum((0.01+(large_rho**2)*0.99)*large_Jelem)  # compliance
        disp_force = large_u[large_FE.loading_point]
        return compliance,disp_force

    def selecting_loading(self, example):
        #  ~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~#
        nelx = self.nelx
        nely = self.nely
        self.example = example
        if(example == 1): # tip cantilever
            self.exampleName = 'TipCantilever'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed = self.dofs[0:2*(nely+1):1] 
            self.force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1 
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular' : None } 
            self.symXAxis = False 
            self.symYAxis = False 
        elif(example == 2): # mid cantilever
            self.exampleName = 'MidCantilever'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed = self.dofs[0:2*(nely+1):1] 
            self.force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1 
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular': None  } 
            self.symXAxis = True 
            self.symYAxis = False 
        elif(example == 3): #  MBBBeam
            self.exampleName = 'MBBBeam'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1) 
            self.force[2*(nely+1)+1 ,0]=-1 
            self.nonDesignRegion =  {'Rect': None, 'Circ' : None, 'Annular': None } 
            self.symXAxis = False 
            self.symYAxis = False 
        elif(example == 4): #  Michell
            self.exampleName = 'Michell'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed=np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] ) 
            self.force[nelx*(nely+1)+1 ,0]=-1 
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular' : {'center':[30.,15.], 'rad_out':6., 'rad_in':3} } 
            self.symXAxis = False 
            self.symYAxis = True 
        elif(example == 5): #  DistributedMBB
            self.exampleName = 'Bridge'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed= np.array([ 0,1,2*(nelx+1)*(nely+1)-2*nely+1,2*(nelx+1)*(nely+1)-2*nely] ) 
            self.force[2*nely+1:2*(nelx+1)*(nely+1):2*(nely+1),0]=-1/(nelx+1) 
            self.nonDesignRegion = {'Rect': {'x>':0, 'x<':nelx,'y>':nely-1,'y<':nely}, 'Circ' : None, 'Annular' : None } 
            self.symXAxis = False 
            self.symYAxis = True 
        elif(example == 6): # Tensile bar
            self.exampleName = 'TensileBar'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof)  
            self.fixed =np.union1d(np.arange(0,2*(nely+1),2), 1)  # fix X dof on left
            self.midDofX= 2*(nelx+1)*(nely+1)- (nely) 
            self.force[self.midDofX, 0 ] = 1 
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular' : None } 
            self.symXAxis = True 
            self.symYAxis = False
        elif(example == 7): # Complex  cantilever
            self.exampleName = 'ComplexCantilever'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed = self.dofs[0:2*(nely+1):1] 
            self.force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1
            self.force[2*(nelx+1)*(nely+1)-2, 0 ] = 1  
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular' : None } 
            self.symXAxis = False 
            self.symYAxis = False
        elif(example == 8): # mid cantilever
            self.exampleName = 'Mid_Complex_Cantilever'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed = self.dofs[0:2*(nely+1):1] 
            self.force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1
            #self.force[2*(nelx+1)*(nely+1)- (nely), 0 ] = 1
            self.force[2*(nelx+1)*(nely+1)-2, 0 ] = 1   
            self.nonDesignRegion = {'Rect': None, 'Circ' : None, 'Annular': None  } 
            self.symXAxis = False 
            self.symYAxis = False
        elif(example == 9): #  MBBBeam
            self.exampleName = 'Complex_MBBBeam'
            self.ndof = 2*(nelx+1)*(nely+1) 
            self.force = np.zeros((self.ndof,1))
            self.dofs=np.arange(self.ndof) 
            self.fixed= np.union1d(np.arange(0,2*(nely+1),2), 2*(nelx+1)*(nely+1)-2*(nely+1)+1) 
            self.force[2*(nely+1)+1 ,0]=-1
            self.force[2*(nelx+1)*(nely+1)-2, 0 ] = 1  
            self.nonDesignRegion =  {'Rect': None, 'Circ' : None, 'Annular': None } 
            self.symXAxis = False 
            self.symYAxis = False   

    def generatePoints(self, nelx, nely, resolution = 1, nonDesignRegion = None): # generate points in elements
        ctr = 0 
        xy = np.zeros((resolution*nelx*resolution*nely,2)) 
        nonDesignIdx = torch.zeros((resolution*nelx*resolution*nely), requires_grad = False).to(device) 
        for i in range(resolution*nelx):
            for j in range(resolution*nely):
                xy[ctr,0] = (i + 0.5)/resolution 
                xy[ctr,1] = (j + 0.5)/resolution 
                if(nonDesignRegion['Rect'] is not None):
                    if( (xy[ctr,0] < nonDesignRegion['Rect']['x<']) and (xy[ctr,0] > nonDesignRegion['Rect']['x>']) and (xy[ctr,1] < nonDesignRegion['Rect']['y<']) and (xy[ctr,1] > nonDesignRegion['Rect']['y>'])):
                        nonDesignIdx[ctr] = 1 
                if(nonDesignRegion['Circ'] is not None):
                    if( ( (xy[ctr,0]-nonDesignRegion['Circ']['center'][0])**2 + (xy[ctr,1]-nonDesignRegion['Circ']['center'][1])**2 ) <= nonDesignRegion['Circ']['rad']**2):
                        nonDesignIdx[ctr] = 1 
                if(nonDesignRegion['Annular'] is not None):
                     locn =  (xy[ctr,0]-nonDesignRegion['Annular']['center'][0])**2 + (xy[ctr,1]-nonDesignRegion['Annular']['center'][1])**2 
                     if ((locn <= nonDesignRegion['Annular']['rad_out']**2) and (locn > nonDesignRegion['Annular']['rad_in']**2) ):
                         nonDesignIdx[ctr] = 1 
                ctr += 1 
        xy = torch.tensor(xy, requires_grad = True).float().view(-1,2).to(device) 
        return xy, nonDesignIdx 