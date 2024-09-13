import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm
import cvxopt 
import cvxopt.cholmod
import sys
if sys.platform == 'linux':
    import torch_sparse_solve 
#-----------------------#

#%%  structural FE
class StructuralFE:
    #-----------------------#
    def initializeSolver(self, nelx, nely, forceBC, fixed, penal = 3,Emin = 1e-3, Emax = 1.0):
        self.Emin = Emin;
        self.Emax = Emax;
        self.penal = penal;
        self.nelx = nelx;
        self.nely = nely;
        self.ndof = 2*(nelx+1)*(nely+1)
        self.KE=self.getDMatrix_torch();
        self.fixed = fixed;
        self.free = np.setdiff1d(np.arange(self.ndof),fixed);
        self.f = forceBC;
        self.edofMat=np.zeros((nelx*nely,8),dtype=int)
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])

        self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()

    def getDMatrix(self):
        E=1
        nu=0.3
        k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
        return (KE)
    def getDMatrix_torch(self):
        E=1
        nu=0.3
        k=torch.tensor([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*torch.tensor([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
        return KE
    
    def getLatticeDMatrix_torch(self,C):
        C11,C12,C13,C22,C23,C33 = C[:,0:1,0:1],C[:,0:1,1:2],C[:,0:1,2:3],C[:,1:2,1:2],C[:,1:2,2:3],C[:,2:3,2:3]
        KE_row0 = torch.cat((C11/3 + C13/2 + C33/3,   C12/4 + C13/3 + C23/3 + C33/4,                 C33/6 - C11/3, C12/4 - C13/3 + C23/6 - C33/4,         - C11/6 - C13/2 - C33/6, - C12/4 - C13/6 - C23/6 - C33/4,                 C11/6 - C33/3, C13/6 - C12/4 - C23/3 + C33/4),dim=2)
        KE_row1 = torch.cat((C12/4 + C13/3 + C23/3 + C33/4,           C22/3 + C23/2 + C33/3, C23/6 - C13/3 - C12/4 + C33/4,                 C22/6 - C33/3, - C12/4 - C13/6 - C23/6 - C33/4,         - C22/6 - C23/2 - C33/6, C12/4 + C13/6 - C23/3 - C33/4,                 C33/6 - C22/3),dim=2)
        KE_row2 = torch.cat((                   C33/6 - C11/3,   C23/6 - C13/3 - C12/4 + C33/4,         C11/3 - C13/2 + C33/3, C13/3 - C12/4 + C23/3 - C33/4,                   C11/6 - C33/3,   C12/4 + C13/6 - C23/3 - C33/4,         C13/2 - C11/6 - C33/6, C12/4 - C13/6 - C23/6 + C33/4),dim=2)
        KE_row3 = torch.cat((   C12/4 - C13/3 + C23/6 - C33/4,                   C22/6 - C33/3, C13/3 - C12/4 + C23/3 - C33/4,         C22/3 - C23/2 + C33/3,   C13/6 - C12/4 - C23/3 + C33/4,                   C33/6 - C22/3, C12/4 - C13/6 - C23/6 + C33/4,         C23/2 - C22/6 - C33/6),dim=2)
        KE_row4 = torch.cat((         - C11/6 - C13/2 - C33/6, - C12/4 - C13/6 - C23/6 - C33/4,                 C11/6 - C33/3, C13/6 - C12/4 - C23/3 + C33/4,           C11/3 + C13/2 + C33/3,   C12/4 + C13/3 + C23/3 + C33/4,                 C33/6 - C11/3, C12/4 - C13/3 + C23/6 - C33/4),dim=2)
        KE_row5 = torch.cat(( - C12/4 - C13/6 - C23/6 - C33/4,         - C22/6 - C23/2 - C33/6, C12/4 + C13/6 - C23/3 - C33/4,                 C33/6 - C22/3,   C12/4 + C13/3 + C23/3 + C33/4,           C22/3 + C23/2 + C33/3, C23/6 - C13/3 - C12/4 + C33/4,                 C22/6 - C33/3),dim=2)
        KE_row6 = torch.cat((                   C11/6 - C33/3,   C12/4 + C13/6 - C23/3 - C33/4,         C13/2 - C11/6 - C33/6, C12/4 - C13/6 - C23/6 + C33/4,                   C33/6 - C11/3,   C23/6 - C13/3 - C12/4 + C33/4,         C11/3 - C13/2 + C33/3, C13/3 - C12/4 + C23/3 - C33/4),dim=2)
        KE_row7 = torch.cat((   C13/6 - C12/4 - C23/3 + C33/4,                   C33/6 - C22/3, C12/4 - C13/6 - C23/6 + C33/4,         C23/2 - C22/6 - C33/6,   C12/4 - C13/3 + C23/6 - C33/4,                   C22/6 - C33/3, C13/3 - C12/4 + C23/3 - C33/4,         C22/3 - C23/2 + C33/3),dim=2)
        KE = torch.cat((KE_row0,KE_row1,KE_row2,KE_row3,KE_row4,KE_row5,KE_row6,KE_row7),dim=1)
        return KE

    #-----------------------#
    def solve(self, density):

        self.densityField = density;
        self.u=np.zeros((self.ndof,1))
        # solve
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = K[self.free,:][:,self.free]
        self.u[self.free,0]=spsolve(K,self.f[self.free,0])

        self.Jelem = (np.dot(self.u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1)

        return self.u, self.Jelem;
    #-----------------------#
    def solvetorch(self,density):
        #density = torch.from_numpy(density)
        self.u = torch.zeros((self.ndof,1))
        KE = self.KE
        f = torch.from_numpy(self.f)
        f = f.type(torch.float)
        temp1 = torch.unsqueeze(torch.flatten(KE),dim=1)
        temp2 = (self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))

        sK=torch.flatten(torch.transpose(temp1*temp2,0,1))

        indices = np.array([self.iK,self.jK])

        K = torch.sparse_coo_tensor(indices, sK, size=(self.ndof,self.ndof)).to_dense()
        if sys.platform == 'linux':
            K_sub = K[self.free,:][:,self.free]
            f_sub = f[self.free,:]
            K_sub = torch.unsqueeze(K_sub,dim=0).to_sparse().double()
            f_sub = torch.unsqueeze(f_sub,dim=0).double()
            res = torch_sparse_solve.solve(K_sub, f_sub)
            res = res.float()
            self.u[self.free,0] = res[0,:,0]
        else:
            self.u[self.free,0] = torch.linalg.solve(K[self.free,:][:,self.free], f[self.free,0])
        self.Jelem = torch.sum((torch.matmul(self.u[self.edofMat].reshape(self.nelx*self.nely,8),KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8)),dim=1)
        return self.u, self.Jelem
    
    def solvetorch_sparse(self,density):
        self.u = torch.zeros((self.ndof,1))
        KE = self.KE
        f = torch.from_numpy(self.f)
        f = f.type(torch.float64)
        temp1 = torch.unsqueeze(torch.flatten(KE),dim=1)
        temp2 = (self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))
        sK=torch.flatten(torch.transpose(temp1*temp2,0,1))

        indices = np.array([self.iK,self.jK])

        K = torch.sparse_coo_tensor(indices, sK, size=(self.ndof,self.ndof),dtype=torch.float64).to_dense()
        K_sub = K[self.free,:][:,self.free]
        f_sub = f[self.free,:]
        K_sub = torch.unsqueeze(K_sub,dim=0).to_sparse()
        f_sub = torch.unsqueeze(f_sub,dim=0)
        res = torch_sparse_solve.solve(K_sub, f_sub)
        res = res.float()
        self.u[self.free,0] = res[0,:,0]
        self.Jelem = torch.sum((torch.matmul(self.u[self.edofMat].reshape(self.nelx*self.nely,8),KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8)),dim=1)
        return self.u, self.Jelem
    
    def solvelatticetorch(self,density,C):
        self.u = torch.zeros((self.ndof,1))
        KE = self.getLatticeDMatrix_torch(C)
        density = torch.unsqueeze(density,dim=1)
        f = torch.from_numpy(self.f)
        f = f.type(torch.float)
        temp1 = torch.flatten(KE,start_dim=1,end_dim=2)
        sK = torch.flatten(temp1*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin)))
        indices = np.array([self.iK,self.jK])
        K = torch.sparse_coo_tensor(indices, sK, size=(self.ndof,self.ndof)).to_dense()
        if sys.platform == 'linux':
            K_sub = K[self.free,:][:,self.free]
            f_sub = f[self.free,:]
            K_sub = torch.unsqueeze(K_sub,dim=0).to_sparse().double()
            f_sub = torch.unsqueeze(f_sub,dim=0).double()
            res = torch_sparse_solve.solve(K_sub, f_sub)
            res = res.float()
            self.u[self.free,0] = res[0,:,0]
        else:
            self.u[self.free,0] = torch.linalg.solve(K[self.free,:][:,self.free], f[self.free,0])
        UK = torch.einsum('ki,kij->kj',self.u[self.edofMat].reshape(self.nelx*self.nely,8),KE)
        self.Jelem = torch.sum(UK*self.u[self.edofMat].reshape(self.nelx*self.nely,8),dim=1)
        return self.u, self.Jelem
    
    def solvelatticetorch_sparse(self,density,C):
        self.u = torch.zeros((self.ndof,1))
        KE = self.getLatticeDMatrix_torch(C)
        density = torch.unsqueeze(density,dim=1)
        f = torch.from_numpy(self.f)
        f = f.type(torch.float)
        temp1 = torch.flatten(KE,start_dim=1,end_dim=2)
        sK = torch.flatten(temp1*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin)))
        indices = np.array([self.iK,self.jK])
        K = torch.sparse_coo_tensor(indices, sK, size=(self.ndof,self.ndof)).to_dense()
        K_sub = K[self.free,:][:,self.free]
        f_sub = f[self.free,:]
        K_sub = torch.unsqueeze(K_sub,dim=0).to_sparse()
        f_sub = torch.unsqueeze(f_sub,dim=0)
        res = torch_sparse_solve.solve(K_sub, f_sub)
        res = res.float()
        self.u[self.free,0] = res[0,:,0]

        #self.u[self.free,0] = torch.linalg.solve(K[self.free,:][:,self.free], f[self.free,0])
        UK = torch.einsum('ki,kij->kj',self.u[self.edofMat].reshape(self.nelx*self.nely,8),KE)
        self.Jelem = torch.sum(UK*self.u[self.edofMat].reshape(self.nelx*self.nely,8),dim=1)
        return self.u, self.Jelem
    


    #-----------------------#
    def solve88(self, density):
        self.densityField = density;
        self.u=np.zeros((self.ndof,1))
        self.KE=self.getDMatrix()
        sK=((self.KE.flatten()[np.newaxis]).T*(self.Emin+(0.01 + density)**self.penal*(self.Emax-self.Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = self.deleterowcol(K,self.fixed,self.fixed).tocoo()
        K = cvxopt.spmatrix(K.data,K.row.astype(np.int32),K.col.astype(np.int32))
        B = cvxopt.matrix(self.f[self.free,0])
        cvxopt.cholmod.linsolve(K,B)
        self.u[self.free,0]=np.array(B)[:,0]

        self.Jelem = (np.dot(self.u[self.edofMat].reshape(self.nelx*self.nely,8),self.KE) * self.u[self.edofMat].reshape(self.nelx*self.nely,8) ).sum(1);
        return self.u, self.Jelem;
    #-----------------------#
    def deleterowcol(self, A, delrow, delcol):
        #Assumes that matrix is in symmetric csc form !
        m = A.shape[0]
        keep = np.delete (np.arange(0, m), delrow)
        A = A[keep, :]
        keep = np.delete (np.arange(0, m), delcol)
        A = A[:, keep]
        return A  
    #-----------------------#
    def plotFE(self):
         #plot FE results
         fig= plt.figure() # figsize=(10,10)
         plt.subplot(1,2,1);
         im = plt.imshow(self.u[1::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         J = ( (self.Emin+self.densityField**self.penal*(self.Emax-self.Emin))*self.Jelem).sum()
         plt.title('U_x , J = {:.2E}'.format(J))
         fig.colorbar(im)
         plt.subplot(1,2,2);
         im = plt.imshow(self.u[0::2].reshape((self.nelx+1,self.nely+1)).T, cmap=cm.jet,interpolation='none')
         fig.colorbar(im)
         plt.title('U_y')
         fig.show()

