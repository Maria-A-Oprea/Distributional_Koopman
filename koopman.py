import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad 
import scipy as sp

class koopman:
    def __init__(self, N, basis):
        """
        Initialized the Koopman operator object, that contains:
        - N = dimension of the approximation
        - basis = a string containing the type of basis functions to be chosen. They can be:
                * Fourier modes
                * Thin plate splines
                * Hermite polynomials 
        - mat = the matrix approximating the Koopman operator. 
        """
        self.dim = N
        self.mat = np.zeros((N, N))
        self.basis = basis
    
    def compute_E_l2(self):
        """
        Given the type of basis functions and their number, it computes the matrix 
        E_ij = \int g_i g_j d\rho
        """
        N = self.dim
        E = np.zeros((N, N))
        
        if self.basis == 'indicators':
            E = np.eye(N)
        return E
      
            
                            
    def compute_matrix(self, data):
        """
        Given the type of basis functions (g_i) and measure data (\mu_j) computes
        Mat_ij = \int g_i d\mu_j
        """
        (K, M) = np.shape(data)
        N = self.dim
        Mat = np.zeros((N, M))
        if self.basis == 'fourier':
            for i in range(N):
                for j in range(M):
                    for k in range(K):
                        Mat[i, j] += (i%2)*np.sin((i + 1)*0.5*data[k, j]) - (i%2 - 1)*np.cos(i/2*data[k, j])
                    Mat[i, j] /= 1/K
        if self.basis == 'indicators':
            for i in range(N):
                for j in range(M):
                    for k in range(K):
                        if i/N <= data[k, j] < (i + 1)/N:
                            Mat[i, j] += 1/K
        return Mat
    
    def L2_dmd(self, data):
        """ implements the DMD algorithm (see section 6.2 in the paper) to compute the matrix C 
                that minimizes || D - EC||_F
            updates self.matrix accordingly
            takes in the data matrix representing the empirical distributions for $\mu_j$ 
        """
        D = self.compute_matrix(data)
        E = self.compute_E_l2()
        print(D)

        self.mat = D@np.linalg.inv(E)
        
    def objective(self, C, E, D):
        N = self.dim 
        C = np.reshape(C, (N, N))
        inner_sup = lambda x:-  np.max(x@(D - C@E))
        constr = [{'type': 'ineq', 'fun': lambda x: 1 - np.linalg.norm(x,ord = 1)}]
        res = sp.optimize.minimize(inner_sup, x0 = 1/N*np.random.rand(N, ),  constraints = constr)
        
        return res.fun
       
    def sup_dmd(self, data0, data1):
        """
            implements the supremum norm optimization i.e. problem (4.17)
            where we take the supremum over \beta and over each column of 
            (D - EC) multiplied on the left  by \beta
        """
        N = self.dim
        D = self.compute_matrix(data1)
        E = self.compute_matrix(data0)
        objective_partial = lambda C:self.objective(C, E=E, D=D)
        res = sp.optimize.minimize(objective_partial, x0 = np.random.rand(N**2, ) ) 
        
        C = np.reshape(res.x, (N, N))
        return C
    
    def sko(self, data):
        """ Performs the classis DMD algorithm on trajectory data. 
            Assumes data = [x_0, x_1, ...., x_M] and computes f_i(x_j) and f_j(x_{i + 1}) to 
                form the matrices E_ij = f_i(x_j) and D_ij = f_i(x_{j + 1}) 
            Computes the Koopman operator by K = E^{-1} D
        """
        N = self.dim
        M = np.shape(data)[0]
        E = np.zeros((N, M - 1))
        D = np.zeros((N, M - 1))
        if self.basis == 'indicators':
            for j in range(M - 1):
                E[int(np.floor(N*data[j])), j] = 1
                D[int(np.floor(N*data[ j + 1 ])), j] = 1
            self.mat = D@np.linalg.pinv(E)
        