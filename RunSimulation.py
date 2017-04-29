from DataDistribution import DataDistribution







mu1 = [[-2.0, 2.0, -5.0, -10.0, 5],
       [3.0, 4.0, -3.0, -7.0, 1.0]]

sd1 = [[1.0, 0.5, 0.5, 0.5, 1.5],
       [0.5, 1.5, 0.5, 1.0, 0.5]]


w1 = [0.2, 0.3, 0.2, 0.1, 0.2]

dist = DataDistribution(mu1, sd1, w1)

x_out = dist.gaussMixture_multivariate([[1],[2]])


