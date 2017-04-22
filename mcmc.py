import numpy as np
import time
import matplotlib.pyplot as plt



class GMM(object):
    def __init__(self, means, sigmas, weights):
        self.means = means
        self.sigmas = sigmas
        self.weights = weights

    def gaussian_mixture(self, x):

        def normpdf(x, mu, sigma):
            u = (x - mu) / abs(sigma)
            y = (1 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)
            return y

        l = len(self.means)
        x_out = 0
        for i in range(l):
            x_out = x_out + self.weights[i] * normpdf(x, self.means[i], self.sigmas[i])
        return x_out


class MCMC(object):
    def __init__(self, x0, sigma, n):
        self.x0 = x0
        self.sigma = sigma
        self.n = n

    def mcmc(self, p: object):

        t = time.time()
        x_old = self.x0
        x = []

        for i in range(self.n):
            x_new = np.random.normal(x_old, self.sigma)
            p_new = p(x_new)
            p_old = p(x_old)
            U = np.random.uniform()

            A = p_new / p_old

            if U < A:
                x_old = x_new

            x.append(x_old)

        print("Elapsed time: %1.2f seconds" % (time.time() - t))

        return x

    def plotHistorgram(self, p, x, xrange, label=u'MCMC distribution'):
        # plot sample histogram
        plt.hist(x, 100, alpha=0.7, label=u'MCMC distribution', normed=True)

        # plot the true function
        xx = np.linspace(xrange[0], xrange[1], 100)
        plt.plot(xx, p(xx), 'r', label=u'True distribution')
        plt.legend()
        plt.show()

        print("Starting point was ", x[0])
