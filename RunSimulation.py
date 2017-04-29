import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import csv



def pair(data, labels=None):

	n_variables = data.shape[1]
	if labels is None:
		labels = ['var%d' % i for i in range(n_variables)]
	fig = plt.figure()
	for i in range(n_variables):
		for j in range(n_variables):
			n_sub = i * n_variables + j + 1
			ax = fig.add_subplot(n_variables, n_variables, n_sub)
			ax.axes.get_xaxis().set_ticklabels([])
			ax.axes.get_yaxis().set_ticklabels([])
			if i == j:
				ax.hist(data[:, i])
				ax.set_title(labels[i])
			else:
				X = np.arange(-25, 25, 0.01)
				Y = np.arange(-25, 25, 0.01)
				X, Y = np.meshgrid(X, Y)
				Z = gaussianMixture(X, Y, pi, centroids, covmat)
				ax.contour(X, Y, Z)
				ax.scatter(data[:, 0], data[:, 1], s=1, color='0.5')
	return fig

def pairs(data):
#"Quick&dirty scatterplot matrix"
    d = len(data)
    fig, axes = plt.subplots(nrows=d, ncols=d, sharex='col', sharey='row')
    for i in range(d):
        for j in range(d):
            ax = axes[i,j]
            if i == j:
                ax.text(0.5, 0.5, transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=16)
            else:
                ax.scatter(data[j], data[i], s=10)


def multivariateGaussianMixture(pi, centroids, covmat, n_samples):
	K = len(pi)
	D = centroids.shape[1]
	N = n_samples
	x_out = np.zeros([N,D])

	pi_cum = np.cumsum(pi)

	for n in range(N):
		u = np.random.uniform()
		for k in range(K):
			if u < pi_cum[0]:
				x_out[n, :] = np.random.multivariate_normal(centroids[0], np.diag(covmat[0]))
			elif u < pi_cum[1]:
				x_out[n, :] = np.random.multivariate_normal(centroids[1], np.diag(covmat[1]))
			elif u < pi_cum[2]:
				x_out[n, :] = np.random.multivariate_normal(centroids[2], np.diag(covmat[2]))
			elif u < pi_cum[3]:
				x_out[n, :] = np.random.multivariate_normal(centroids[3], np.diag(covmat[3]))
			elif u < pi_cum[4]:
				x_out[n, :] = np.random.multivariate_normal(centroids[4], np.diag(covmat[4]))
			elif u < pi_cum[5]:
				x_out[n, :] = np.random.multivariate_normal(centroids[5], np.diag(covmat[5]))
			else:
				x_out[n, :] = np.random.multivariate_normal(centroids[6], np.diag(covmat[6]))
	return x_out

def gaussianMixture(X, Y, pi, centroids, covmat):
	K = len(pi)
	D = centroids.shape[1]
	N = len(X)
	mix = np.zeros([N, N])
	for k in range(K):
		mix = mix + pi[k] * mlab.bivariate_normal(X, Y, covmat[k, 0], covmat[k, 1], centroids[k, 0], centroids[k, 1], 1)
	return mix


"""
Multivariate Gaussian mixture
"""
pi = np.array([12, 24, 32, 10, 15, 29, 100])
pi = pi/pi.sum()


np.random.uniform(-15, 15, 50)
centroids = np.array([[0, 0, 0, 0, 0],
					 [-10, 5, 9, 0, 0],
					 [-5, -1, 3, 4, -7],
					 [4, -3, 4, 2, 1],
					 [8, 5, 6, 14, -4],
					 [2, 2, -9, -5, 6],
					 [12, -1, 4, 0, 12]])

covmat = np.tile([3, 4, 5, 3, 1], (7,1))

sim_out = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 50)

pairs(sim_out)


"""
Bivariate Gaussian mixture
"""
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


sim_out = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 500)

pair(sim_out)


X = np.arange(-25, 25, 0.01)
Y = np.arange(-25, 25, 0.01)
X, Y = np.meshgrid(X, Y)
Z = gaussianMixture(X, Y, pi, centroids, covmat)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap= plt.get_cmap("coolwarm"), linewidth=0, antialiased=False, shade = True, alpha = 0.7)
ax.view_init(50, 100)


# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(plt.LinearLocator(10))
ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


plt.contour(X, Y, Z)
plt.scatter(sim_out[:,0], sim_out[:,1], s=1,color='0.5')

sim_out_500 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 500)
np.savetxt("bivariate_simulation_n500.csv", sim_out_500, delimiter=",")

sim_out_1000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 1000)
np.savetxt("bivariate_simulation_n1000.csv", sim_out_1000, delimiter=",")

sim_out_5000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 5000)
np.savetxt("bivariate_simulation_n5000.csv", sim_out_5000, delimiter=",")

sim_out_10000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 100000)
np.savetxt("bivariate_simulation_n100000.csv", sim_out_10000, delimiter=",")

mm2_mean = np.dot(pi, centroids)
np.savetxt("bivariate50D_TrueMeans.csv", mm2_mean, delimiter=",")
np.savetxt("bivariate50D_TruePi.csv", pi, delimiter=",")
np.savetxt("bivariate50D_TrueCentroids.csv", centroids, delimiter=",")
np.savetxt("bivariate50D_TrueCov.csv", covmat, delimiter=",")





sim_out_1000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 1000)
np.savetxt("multivariate5D_simulation_n1000.csv", sim_out_1000, delimiter=",")

sim_out_5000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 5000)
np.savetxt("multivariate5D_simulation_n5000.csv", sim_out_5000, delimiter=",")

sim_out_10000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 10000)
np.savetxt("multivariate5D_simulation_n10000.csv", sim_out_10000, delimiter=",")

sim_out_50000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 50000)
np.savetxt("multivariate5D_simulation_n50000.csv", sim_out_50000, delimiter=",")

mm50_mean = np.dot(pi, centroids)
np.savetxt("multivariate5D_TrueMeans.csv", mm50_mean, delimiter=",")
np.savetxt("multivariate5D_TruePi.csv", pi, delimiter=",")
np.savetxt("multivariate5D_TrueCentroids.csv", centroids, delimiter=",")
np.savetxt("multivariate5D_TrueCov.csv", covmat, delimiter=",")




sim_out_5000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 5000)
np.savetxt("multivariate50D_simulation_n5000.csv", sim_out_5000, delimiter=",")

sim_out_10000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 10000)
np.savetxt("multivariate50D_simulation_n10000.csv", sim_out_10000, delimiter=",")

sim_out_50000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 50000)
np.savetxt("multivariate50D_simulation_n50000.csv", sim_out_50000, delimiter=",")

sim_out_100000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 100000)
np.savetxt("multivariate50D_simulation_n100000.csv", sim_out_50000, delimiter=",")

sim_out_1000000 = multivariateGaussianMixture(pi, centroids, covmat, n_samples = 1000000)
np.savetxt("multivariate50D_simulation_n1000000.csv", sim_out_50000, delimiter=",")

mm50_mean = np.dot(pi, centroids)
np.savetxt("multivariate50D_TrueMeans.csv", mm50_mean, delimiter=",")
np.savetxt("multivariate50D_TruePi.csv", pi, delimiter=",")
np.savetxt("multivariate50D_TrueCentroids.csv", centroids, delimiter=",")
np.savetxt("multivariate50D_TrueCov.csv", covmat, delimiter=",")



var50_mean = np.dot(np.power(pi,2), covmat)