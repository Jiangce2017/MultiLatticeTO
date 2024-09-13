import torch.nn as nn
import torch
from utils import set_seed
from fourier_2d import FNO2d,FNO2d_Lattice,CNN2d_Lattice

#%% Neural network
class TopNet(nn.Module):
    # inputDim = 2; # x and y coordn of the point
    # outputDim = 2; # if material/void at the point
    
    def __init__(self, config,symXAxis,symYAxis):
        self.nelx = config.nelx; # to impose symm, get size of domain
        self.nely = config.nely;
        self.inputDim = 2
        if config.searchMode == 'simplex':
            self.outputDim = 2 + config.simplexDim
        elif config.searchMode == 'cubic':
            self.outputDim = 1 + config.latentDim
        self.searchMode = config.searchMode
        self.simplexDim = config.simplexDim
        self.latentDim = config.latentDim 
        self.symXAxis = symXAxis; # set T/F to impose symm
        self.symYAxis = symYAxis;
        super().__init__();
        self.layers = nn.ModuleList();
        current_dim = self.inputDim;
        manualSeed = 1234; # NN are seeded manually 
        set_seed(manualSeed)
        for lyr in range(config.numLayers): # define the layers
            l = nn.Linear(current_dim, config.numNeuronsPerLyr);
            nn.init.xavier_normal_(l.weight);
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = config.numNeuronsPerLyr;
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(config.numLayers): # batch norm 
            self.bnLayer.append(nn.BatchNorm1d(config.numNeuronsPerLyr));
    def forward(self, x,resolution, fixedIdx = None):
        # LeakyReLU ReLU6 ReLU
        m = nn.ReLU6(); # LeakyReLU 
        ctr = 0
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx);
        else:
            xv = x[:,0];
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) ;
        else:
            yv = x[:,1];
        x = torch.transpose(torch.stack((xv,yv)),0,1)

        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)))
            ctr += 1
        x = self.layers[-1](x)
        out = x.view(-1,self.outputDim)
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            #print("out shape: {}".format(out.shape))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]
        return  rho, t
        
    def  getWeights(self): # stats about the NN
        modelWeights = [];
        modelBiases = [];
        for lyr in self.layers:
            modelWeights.extend(lyr.weight.data.view(-1).cpu().numpy());
            modelBiases.extend(lyr.bias.data.view(-1).cpu().numpy());
        return modelWeights, modelBiases;

class TopFNO(nn.Module):
    #inputDim = 2; # x and y coordn of the point
    #outputDim = 2; # if material/void at the point
    def __init__(self,config,symXAxis,symYAxis):
        self.nelx = config.nelx; # to impose symm, get size of domain
        self.nely = config.nely;
        self.inputDim = 2
        if config.searchMode == 'simplex':
            self.outputDim = 2 + config.simplexDim
        elif config.searchMode == 'cubic':
            self.outputDim = 1 + config.latentDim
        self.searchMode = config.searchMode
        self.simplexDim = config.simplexDim
        self.latentDim = config.latentDim
        #self.outputDim = 1+ latentParDim
        self.symXAxis = symXAxis; # set T/F to impose symm
        self.symYAxis = symYAxis;
        super().__init__();
        manualSeed = 1234; # NN are seeded manually 
        set_seed(manualSeed)
        # self.model = FNO(n_modes=(numModex,numModey), hidden_channels=64,in_channels=self.inputDim,out_channels=self.outputDim,n_layers=4,\
        #                  factorization='tucker', rank=0.42,non_linearity=nn.ReLU())
        
        #self.model = FNO2d(numModex,numModey,numNeuronsPerLyr)
        self.model = FNO2d_Lattice(config.numLayers,config.numModex,config.numModey,config.numNeuronsPerLyr,config.searchMode,config.simplexDim,config.latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx);
        else:
            xv = x[:,0];
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)

        x = self.model(x) 

        if (self.symYAxis):
            x_mid_idx = self.nelx//2
            x_back = torch.flip(x[:,:y_mid_idx,:,:],dims=(1,))
            if self.nelx % 2 == 0:
                x[:,x_mid_idx:,:,:] = x_back
            else:
                x[:,x_mid_idx+1:,:,:] = x_back
        if (self.symXAxis):
            y_mid_idx = self.nely//2
            y_back = torch.flip(x[:,:,:y_mid_idx,:],dims=(2,))
            if self.nely % 2 == 0:
                x[:,:,y_mid_idx:,:] = y_back
            else:
                x[:,:,y_mid_idx+1:,:] = y_back

        out = x.view(-1,self.outputDim)
        #print("out shape: {}".format(out.shape))
        #print("fixedIdx shape: {}".format(fixedIdx.shape))
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]

        # out = torch.sigmoid(x)
        # out = out.view(-1,self.outputDim)
        # rho = out[:,0] # grab only the first output
        # rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        # t = out[:,1:]
        return  rho, t
    
    def  getWeights(self): # stats about the NN
        modelWeights = [];
        modelBiases = [];
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy());
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy());
        return modelWeights, modelBiases;

class TopCNN(nn.Module):
    #inputDim = 2; # x and y coordn of the point
    #outputDim = 2; # if material/void at the point
    def __init__(self,numLayers, numModex,numModey,numNeuronsPerLyr,nelx, nely, symXAxis, symYAxis,searchMode,simplexDim,latentDim):
        self.nelx = nelx; # to impose symm, get size of domain
        self.nely = nely;
        self.inputDim = 2
        if searchMode == 'simplex':
            self.outputDim = 2 + simplexDim
        elif searchMode == 'cubic':
            self.outputDim = 1 + latentDim
        self.searchMode = searchMode
        self.simplexDim = simplexDim
        self.latentDim = latentDim
        #self.outputDim = 1+ latentParDim
        self.symXAxis = symXAxis; # set T/F to impose symm
        self.symYAxis = symYAxis;
        super().__init__();
        manualSeed = 1234; # NN are seeded manually 
        set_seed(manualSeed)
        # self.model = FNO(n_modes=(numModex,numModey), hidden_channels=64,in_channels=self.inputDim,out_channels=self.outputDim,n_layers=4,\
        #                  factorization='tucker', rank=0.42,non_linearity=nn.ReLU())
        
        #self.model = FNO2d(numModex,numModey,numNeuronsPerLyr)
        self.model = CNN2d_Lattice(numLayers,numModex,numModey,numNeuronsPerLyr,searchMode,simplexDim,latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx);
        else:
            xv = x[:,0];
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)

        x = self.model(x)  

        out = x.view(-1,self.outputDim)
        #print("out shape: {}".format(out.shape))
        #print("fixedIdx shape: {}".format(fixedIdx.shape))
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]

        # out = torch.sigmoid(x)
        # out = out.view(-1,self.outputDim)
        # rho = out[:,0] # grab only the first output
        # rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        # t = out[:,1:]
        return  rho, t
    
    def  getWeights(self): # stats about the NN
        modelWeights = [];
        modelBiases = [];
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy());
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy());
        return modelWeights, modelBiases;

class TopFNO_branch(nn.Module):
    #inputDim = 2; # x and y coordn of the point
    #outputDim = 2; # if material/void at the point
    def __init__(self,numLayers, numModex,numModey,numNeuronsPerLyr,nelx, nely, symXAxis, symYAxis,searchMode,simplexDim,latentDim):
        self.nelx = nelx; # to impose symm, get size of domain
        self.nely = nely;
        self.inputDim = 2
        if searchMode == 'simplex':
            self.outputDim = 2 + simplexDim
        elif searchMode == 'cubic':
            self.outputDim = 1 + latentDim
        self.searchMode = searchMode
        self.simplexDim = simplexDim
        self.latentDim = latentDim
        #self.outputDim = 1+ latentParDim
        self.symXAxis = symXAxis; # set T/F to impose symm
        self.symYAxis = symYAxis;
        super().__init__();
        manualSeed = 1234; # NN are seeded manually 
        set_seed(manualSeed)
        # self.model = FNO(n_modes=(numModex,numModey), hidden_channels=64,in_channels=self.inputDim,out_channels=self.outputDim,n_layers=4,\
        #                  factorization='tucker', rank=0.42,non_linearity=nn.ReLU())
        
        #self.model = FNO2d(numModex,numModey,numNeuronsPerLyr)
        self.lattice_param_model = FNO2d_Lattice(numLayers,numModex,numModey,10,searchMode,simplexDim-1,latentDim)
        self.density_model = FNO2d_Lattice(numLayers,2,1,5,searchMode,-1,latentDim)
    def forward(self, x,resolution, fixedIdx = None):
        if(self.symYAxis):
            xv = 0.5*self.nelx + torch.abs( x[:,0] - 0.5*self.nelx);
        else:
            xv = x[:,0];
        if(self.symXAxis):
            yv = 0.5*self.nely + torch.abs( x[:,1] - 0.5*self.nely) 
        else:
            yv = x[:,1]
        x = torch.transpose(torch.stack((xv,yv)),0,1)
        
        x = x.view(1,self.nelx*resolution,self.nely*resolution,2)

        density = self.density_model(x) 
        lattice_param = self.lattice_param_model(x)
        x = torch.cat((density,lattice_param),dim=-1)

        if (self.symYAxis):
            x_mid_idx = self.nelx//2
            x_back = torch.flip(x[:,:y_mid_idx,:,:],dims=(1,))
            if self.nelx % 2 == 0:
                x[:,x_mid_idx:,:,:] = x_back
            else:
                x[:,x_mid_idx+1:,:,:] = x_back
        if (self.symXAxis):
            y_mid_idx = self.nely//2
            y_back = torch.flip(x[:,:,:y_mid_idx,:],dims=(2,))
            if self.nely % 2 == 0:
                x[:,:,y_mid_idx:,:] = y_back
            else:
                x[:,:,y_mid_idx+1:,:] = y_back

        out = x.view(-1,self.outputDim)
        #print("out shape: {}".format(out.shape))
        #print("fixedIdx shape: {}".format(fixedIdx.shape))
        if self.searchMode == 'simplex':
            rho = torch.sigmoid(out[:,0])
            rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
            t = torch.softmax(out[:,1:],dim=1)
        elif self.searchMode == 'cubic':
            out = torch.sigmoid(out)
            rho = out[:,0]
            t = out[:,1:]

        # out = torch.sigmoid(x)
        # out = out.view(-1,self.outputDim)
        # rho = out[:,0] # grab only the first output
        # rho = (1-fixedIdx)*rho + fixedIdx*(rho + torch.abs(1-rho))
        # t = out[:,1:]
        return  rho, t
    
    def  getWeights(self): # stats about the NN
        modelWeights = [];
        modelBiases = [];
        modelWeights.extend(self.model.weight.data.view(-1).cpu().numpy());
        modelBiases.extend(self.model.bias.data.view(-1).cpu().numpy());
        return modelWeights, modelBiases;
