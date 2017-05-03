import numpy as np
from GaussianMixture import GaussianMixture


# ********************************************
# inpute parameters below
# ********************************************
pi = np.array([20, 34, 32, 45, 40, 29, 70])
pi = pi/pi.sum()

centroids = np.array([[0, 0],
					 [-15, 5],
					 [-5, -9],
					 [4, -3],
					 [8, 12],
					 [2, 2],
					 [15, -1]])

covmat = np.array([[3, 4],
					 [3, 2.5],
					 [2, 3],
					 [14, 2],
					 [4, 3],
					 [3, 4],
					 [4, 13]])

# ********************************************
# Initialize GaussianMixture class
# ********************************************
gm = GaussianMixture(pi, centroids, covmat)

# ********************************************
# Simulate from Gaussian mixture
# ********************************************
gm_sim = gm.simulate_gm(n_samples=100)

# ********************************************
# Obtain pdf of mixture for some input X
# ********************************************
pdf = gm.pdf_gm(gm_sim)


# ********************************************
# Visualize pdf of mixture in bivariate case
# ********************************************
gm.surface_gm(ranges=[-25,25], fineness=.5, filename="surfacelot.png")


# ********************************************
# Visualize pdf of mixture in bivariate case
# ********************************************
gm.pairs_gm(gm_sim, filename="pairsplot.png", ranges=[-25,25])
