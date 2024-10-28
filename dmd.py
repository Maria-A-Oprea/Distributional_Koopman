import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as skd

class DMD:
    def __init__(self, n1, n2):
        self.eval = np.zeros(n2)
        self.eivect = np.zeros((n2, n2))
        self.mat = np.zeros((n2, n2))
    
    def Algo1(self, X):
        [m, T] = np.shape(X)
        XX = X[:, :-1]
        XY = X[:, 1:]
        C = np.matmul(XY, np.linalg.pinv(XX, etol = ))
        self.eval, self.eivect = np.linalg.eig(C)
        self.matrix = C

    def Plot_evals(self, circle = False):
        # plots eigenvalues of the approximated koopman operator
        # and draws the unit circle if the argument circle is True

        reals = np.real(self.eval)
        imgs = np.imag(self.eval)
        tol = 1e3
        if np.any(reals) > tol:
            print("Warning: Eigenvalue real part is greater than tolerance")
            evals = self.eval(reals < tol and imgs < tol)
        if np.any(imgs) > tol:
            print("Warning: Eigenvalue imaginary part is greater than tolerance")
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.plot(reals, imgs, 'x', label = "dmd_evals")
        if circle == True:
            plt.plot(np.linspace(-1, 1, 100), np.sqrt(1 - np.linspace(-1, 1, 100)**2), 'k')
            plt.plot(np.linspace(-1, 1, 100), - np.sqrt(1 - np.linspace(-1, 1, 100)**2), 'k')
        plt.grid()
    

    def predict(self, X):
        # given an input vector X, it predicts the state X at dt = 1e-3 into the future
        return self.matrix @ X
