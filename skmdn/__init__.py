from sklearn.base import BaseEstimator
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

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
    '''
    def __init__(self, hidden_dim=10, n_gaussians=5, epochs=1000, lr=0.01):
        self.hidden_dim = hidden_dim
        self.n_gaussians = n_gaussians
        self.epochs = epochs
        self.lr = lr
    
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
        self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()

        return self
    
    def partial_fit(self, X, y, n_epochs=1):
        """
        Fit the model to the data for a few epochs.
        
        Args:
            X: (n_samples, n_features)
            y: (n_samples, 1)
            n_epochs: number of epochs
        """
        X, y = self._cast_torch(X, y)

        if not self.optimizer_:
            self.optimizer_ = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        for epoch in range(n_epochs):
            self.optimizer_.zero_grad()
            pi, mu, sigma = self.model_(X)
            loss = mdn_loss(pi, mu, sigma, y)
            loss.backward()
            self.optimizer_.step()

        return self
    
    def forward(self, X):
        """
        Predict the mean of the distribution for each sample in X.
        
        Args:
            X: (n_samples, n_features)
        
        Returns:
            y: (n_samples,)
        """
        X = torch.tensor(X, dtype=torch.float32).clone().detach()
        with torch.no_grad():
            pi, mu, sigma = self.model_(X)
        pi, mu, sigma = pi.detach().numpy(), mu.detach().numpy(), sigma.detach().numpy()
        return pi, mu[:, :, 0], sigma[:, :, 0]

    def pdf(self, X, resolution=100):
        '''
        Compute the probability density function of the model.

        This function computes the pdf for each sample in X, returning a pdf for each sample. 
        It also returns the y values for which the pdf is computed to help with plotting.

        Args:
            X: (n_samples, n_features)
            resolution: number of points in the output pdf

        Returns:
            pdf: (n_samples, resolution)
            ys: (resolution,)
        '''
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pi, mu, sigma = self.model_(X)
        pi, mu, sigma = self.forward(X)
        ys = np.linspace(self.y_min_, self.y_max_, resolution)
        pdf = np.zeros((pi.shape[0], resolution))
        for i in range(pi.shape[0]):
            for j in range(pi.shape[1]):
                pdf[i] += norm(mu[i, j], sigma[i, j]).pdf(ys) * pi[i, j]
        return pdf, ys
    
    def predict_var(self, X, quantile=0.9, resolution=100):
        '''
        Predicts the variance at risk at a given quantile for each datapoint X.
        
        Args:
            X: (n_samples, n_features)
            quantile: quantile value

        Returns:
            risk: (n_samples,)
        '''
        pdf, ys = self.pdf(X, resolution=resolution)
        cdf = pdf.cumsum(axis=1)
        cdf /= cdf[:, -1].reshape(-1, 1)
        risk = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            risk[i] = ys[np.argmax(cdf[i] > quantile)]
        return risk