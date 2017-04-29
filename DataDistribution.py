import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal


class DataDistribution(object):
    def __init__(self, means, sigmas, weights):
        """
        Initialize class variables
        :param means: vector of means of dimension d for each distribution in mixture
        :param sigmas: vector of variances of dimension d for each distribution in mixture
        :param weights: vector of weights of dimension d for each distribution in mixture
        """
        self.means = means
        self.sigmas = sigmas
        self.weights = weights

    def gaussMixture_univariate(self, x):
        """
        Generates a Gaussian mixture distribution
        :param x: the quantile of the gaussian with which to sample the pdf
        :return: x_out returns the probability density from the quantile x
        """
        def normpdf(x, mu, sigma):
            """
            The pdf of the normal distribution
            :param x: quantile on which to sample the density function
            :param mu: scalar vector of mean of each of the Gaussians
            :param sigma: scalar vector of variance of each of the Gaussians
            :return: the value of the pdf for a simple Gaussian
            """
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)
            return y
        l = len(self.means)
        x_out = 0

        # build the mixture via a weights combination of normpdf
        for i in range(l):
            x_out = x_out + self.weights[i] * normpdf(x, self.means[i], self.sigmas[i])
        return x_out

    def gaussMixture_multivariate(self, x):
        d = np.shape(self.means)[0]
        k = np.shape(self.means)[1]
        x_out = np.zeros([1,d])
        x_in = np.transpose(x)

        means = np.transpose(self.means)
        cov = np.transpose(self.sigmas)

        for i in range(k):
            x_out[0,:] = x_out[0,:] + self.weights[i] * multivariate_normal.pdf(x, means[i,:], np.diag(cov[i,:]))
        return x_out
