import numpy as np
from GaussianMixture import GaussianMixture
from scipy.stats import multivariate_normal as mvnorm



pi = np.array([20, 34, 32, 45, 40, 29, 70])
pi = pi/pi.sum()

centroids = np.array([[0, 0],
					 [-15, 5],
					 [-5, -9],
					 [4, -3],
					 [8, 12],
					 [2, 2],
					 [15, -1]])

covmat = np.tile([3, 4], (7,1))

covmat = np.array([[3, 4],
					 [3, 2.5],
					 [2, 3],
					 [14, 2],
					 [4, 3],
					 [3, 4],
					 [4, 13]])

X = np.arange(-25, 25, 0.5)
Y = np.arange(-25, 25, 0.5)
X,Y = np.meshgrid(X, Y)
grid = np.stack([X,Y])

gm = GaussianMixture(pi, centroids, covmat)
gm_sim = gm.simulate_gm(100)
pdf = gm.pdf_gm(gm_sim[1,:])
#gm.pairs_gm(gm_sim)
gm.surface_gm([-25,25], 0.5, filename="testplot.png")