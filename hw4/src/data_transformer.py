from abc import ABC, abstractmethod

class DataTransformer(ABC):
    @abstractmethod
    def fit(self, X):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        None
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """Apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        pass

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)