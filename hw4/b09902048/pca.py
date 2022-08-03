from data_transformer import DataTransformer
import numpy as np
from numpy import matlib


class PCA(DataTransformer):
    """
    Add more of your code here if you want to
    Parameters
    ----------
    n_components : int
        Number of components to keep.
    """
    

    def __init__(self, args, U=None):
        self.n_components = args.latent_space_dim
        self.U = U

    def fit(self, X):
        N= np.size(X, 0)
        def zero_mean(data):
            avg = np.average(data)
            return data - matlib.repmat(avg, N, 1)

        # get covariance matrix
        x = zero_mean(X)
        Sigma = np.dot(np.transpose(x), x) / N
        
        # get svd
        U, S, V = np.linalg.svd(Sigma)
        self.U = U
        return U[:, 0:4].T
        #raise NotImplementedError
    
    def transform(self, X):
        U_reduce = self.U[:, 0:self.n_components]
        Z = X @ U_reduce
        return Z
        raise NotImplementedError

    
    def reconstruct(self, X_transformed):
        """
        Reconstruct the transformed X
        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_components)
            of reconstructed values.
        """
        U_trans = np.transpose(self.U[:,0:self.n_components])
        return X_transformed @ U_trans
        raise NotImplementedError
