from sklearn.base import BaseEstimator
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MixtureDensityNetwork(nn.Module):
    '''
    A simple Mixture Density Network that predicts a distribution over a single output variable.
    
    Args:
        input_dim: input dimension
        hidden_dim: hidden layer dimension
        output_dim: output dimension
        n_gaussians: number of gaussians in the mixture model
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, n_gaussians):
        super(MixtureDensityNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.pi_layer = nn.Linear(hidden_dim, n_gaussians)
        self.mu_layer = nn.Linear(hidden_dim, n_gaussians * output_dim)
        self.sigma_layer = nn.Linear(hidden_dim, n_gaussians * output_dim)

    def forward(self, x):
        h = F.softplus(self.hidden_layer(x))
        pi = F.softmax(self.pi_layer(h), dim=1)
        mu = self.mu_layer(h).view(-1, self.n_gaussians, self.output_dim)
        sigma = F.softplus(self.sigma_layer(h)).view(-1, self.n_gaussians, self.output_dim)
        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, target):
    """
    MDN Loss Function
    
    Args:
        pi: (batch_size, n_gaussians)
        mu: (batch_size, n_gaussians)
        sigma: (batch_size, n_gaussians)
        target: (batch_size, 1)
    
    Returns:
        loss: scalar
    """
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target.unsqueeze(1).expand_as(mu))
    weighted_logprob = log_prob + torch.log(pi.unsqueeze(-1))
    return -torch.logsumexp(weighted_logprob, dim=1).mean()


class MixtureDensityEstimator(BaseEstimator):
    '''
    A scikit-learn compatible Mixture Density Estimator.
    
    Args:
        hidden_dim: hidden layer dimension
        n_gaussians: number of gaussians in the mixture model
        epochs: number of epochs
        lr: learning rate
        weight_decay: weight decay for regularisation
    '''
    def __init__(self, hidden_dim=10, n_gaussians=5, epochs=1000, lr=0.01, weight_decay=0.0):
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
    
    def _cast_torch(self, X, y):
        if not hasattr(self, 'X_width_'):
            self.X_width_ = X.shape[1]
        if not hasattr(self, 'X_min_:'):
            self.X_min_ = X.min(axis=0)
        if not hasattr(self, 'X_max_:'):
            self.X_max_ = X.max(axis=0)
        if not hasattr(self, 'y_min_:'):
            self.y_min_ = y.min()
        if not hasattr(self, 'y_max_:'):
            self.y_max_ = y.max()
        
        assert X.shape[1] == self.X_width_, "Input dimension mismatch"

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples, 1)
        """
        X, y = self._cast_torch(X, y)

        self.model_ = MixtureDensityNetwork(X.shape[1], self.hidden_dim, y.shape[1], self.n_gaussians)
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()

        return self
    
    def partial_fit(self, X, y, n_epochs=1):
        """
        Fit the model to the data for a set number of epochs. Can be used to continue training on new data.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples, 1)
            n_epochs: number of epochs
        """
        X, y = self._cast_torch(X, y)

        if not self.optimizer_:
            self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(n_epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()

        return self
    
    def forward(self, X):
        """
        Calculate the $\pi$, $\mu$ and $\sigma$ outputs n for each sample in X.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            pi: (n_samples, n_gaussians)
            mu: (n_samples, n_gaussians)
            sigma: (n_samples, n_gaussians)
        """
        X = torch.FloatTensor(X)
        with torch.no_grad():
            pi, mu, sigma = self.model_(X)
        pi, mu, sigma = pi.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()
        return pi, mu[:, :, 0], sigma[:, :, 0]

    def pdf(self, X, resolution=100, y_min=None, y_max=None):
        '''
        Compute the probability density function of the model.

        This function computes the pdf for each sample in X.
        It also returns the y values for which the pdf is computed to help with plotting.

        Args:
            X: (n_samples, n_features)
            resolution: number of intervals to compute the quantile over

        Returns:
            pdf: (n_samples, resolution)
            ys: (resolution,)
        '''
        X = torch.FloatTensor(X)
        pi, mu, sigma = self.forward(X)

        ys = np.linspace(y_min or self.y_min_, y_max or self.y_max_, resolution)
        ys_broadcasted = np.broadcast_to(ys, (pi.shape[1], pi.shape[0], resolution)).T
        pdf = np.sum(norm(mu, sigma).pdf(ys_broadcasted) * np.float64(pi), axis=2).T
        return pdf, ys

    def cdf(self, X, resolution=100):
        '''
        Compute the cumulative probability density function of the model.

        This function computes the cdf for each sample in Xd.
        It also returns the y values for which the cdf is computed to help with plotting.

        Args:
            X: (n_samples, n_features)
            resolution: number of intervals to compute the quantile over

        Returns:
            cdf: (n_samples, resolution)
            ys: (resolution,)
        '''
        pdf, ys = self.pdf(X, resolution=resolution)
        cdf = pdf.cumsum(axis=1)
        cdf /= cdf[:, -1].reshape(-1, 1)
        return cdf, ys
        
    def predict(self, X, quantiles=None, resolution=100):
        '''
        Predicts the variance at risk at a given quantile for each datapoint X.
        
        Args:
            X: (n_samples, n_features)
            quantile: quantile value
            resolution: number of intervals to compute the quantile over

        Returns:
            pred: (n_samples,)
            quantiles: (n_samples, n_quantiles)
        '''
        cdf, ys = self.cdf(X, resolution=resolution)
        
        mean_pred = ys[np.argmax(cdf > 0.5, axis=1)]
        
        if not quantiles:
            return mean_pred
        
        quantile_out = np.zeros((X.shape[0], len(quantiles)))
        for j, q in enumerate(quantiles):
            quantile_out[:, j] = ys[np.argmax(cdf > q, axis=1)]
        return mean_pred, quantile_out
