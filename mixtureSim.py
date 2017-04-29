import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as dmvnorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.mlab import bivariate_normal



# Mixing coefficients
pi = [0.4, 0.2, 0.4]

# centroids
centroids = [ np.array([[0],[1]]), np.array([[4],[3]]), np.array([[2],[6]]) ]

# Covariance matrices
cov = [ np.eye(2), np.eye(2), np.eye(2) ]

def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))


def gaussMixtureMap(x, y, pi, centroids, cov):
    d = np.shape(centroids)[1]
    K = np.shape(centroids)[0]
    N = len(x)
    xw = 0
    x_out = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            xyRowvec = np.array([[x[i], y[j]]]).transpose()
            for k in range(K):
                xw = xw + pi[k] * pdf_multivariate_gauss(xyRowvec, centroids[k], cov[k])
            x_out[i,j] = xw
    return x_out

def simGM(pi, centroids, cov, N):
    pi_cumSum = np.cumsum(pi)
    d = np.shape(centroids)[1]
    gSim = np.zeros((N,d))
    for n in range(N):
        r = np.random.random()
        if r < pi_cumSum[0]:
            gSim[n, :] = np.random.multivariate_normal(centroids[0], cov[0])
        elif r < pi_cumSum[1]:
            gSim[n, :] = np.random.multivariate_normal(centroids[1], cov[1])
        else:
            gSim[n, :] = np.random.multivariate_normal(centroids[2], cov[2])
    return gSim



sim_out = simGM(pi, centroids, cov, 1000)



for i in range(np.shape(sim_out)[0]):
    out = gaussMixture_multivariate(sim_out[i, :], pi, centroids, cov)
    print(out)
out = gaussMixture_multivariate(sim_out, pi, centroids, cov)

plt.plot(sim_out[:,0], sim_out[:,1], '.')


x = np.arange(0, 10, 0.1)
y = np.arange(0, 10, 0.1)
z = gaussMixtureMap(x,y, pi, centroids, cov)
plt.contourf(x, y, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)

