import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(42)

samples, labels = make_blobs(n_samples=15000000, centers=10, random_state=0)
k_means = KMeans(10, precompute_distances=True)
k_means.fit(samples)
