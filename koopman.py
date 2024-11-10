import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad 

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
            E =np.eye(N)
        return E
      
            
    def compute_E(self, data):
        """ 
        Given the data $\pi_j$ and the basis vectors g_i computes 
        E_ij = \int g_i d\pi_j
        """
        if self.basis == 'fourier':
            for i in range(N):
                for j in range(N):
                    f = lambda x:  ((i%2)*np.sin((i + 1)/2*x) - 
                                    (i%2 - 1)*np.cos(i/2*x))*((j%2)*np.sin((j + 1)/2*x) - (j%2 - 1)* np.cos(j/2*x))
                    E[i, j] = quad(f, 0, 2*np.pi)[0]
        pass
    def compute_D(self, data):
        """
        Given the type of basis functions and the data (\pi_j, \mu_j) compute
        D_ij = \int g_i d\mu_j
        """
        (K, N) = np.shape(data)
        D = np.zeros((N, N))
        if self.basis == 'fourier':
            for i in range(N):
                for j in range(N):
                    for k in range(K):
                        D[i, j] += (i%2)*np.sin((i + 1)*0.5*data[k, j]) - (i%2 - 1)*np.cos(i/2*data[k, j])
                    D[i, j] /= 1/K
        if self.basis == 'indicators':
            for i in range(N):
                for j in range(N):
                    for k in range(K):
                        if i/N <= data[k, j] < (i + 1)/N:
                            D[i, j] += 1/K
        return D
    
    def L2_dmd(self, data):
        """ implements the DMD algorithm (see section 6.2 in the paper) to compute the matrix C 
                that minimizes || D - EC||_F
            updates self.matrix accordingly
            takes in the data matrix representing the empirical distributions for $\mu_j$ 
        """
        D = self.compute_D(data)
        E = self.compute_E_l2()
        print(D)
        print(E)

        self.mat = D@np.linalg.inv(E)
       
    def sup_dmd(self ):
        pass
    
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
        print(E, D)
