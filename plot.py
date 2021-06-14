import matplotlib.pyplot as plt
import numpy as np

data_list = np.load("data_12.npy")
plt.scatter(data_list[:, 0], data_list[:, 2])
plt.scatter(data_list[:, 0], data_list[:, 1], marker="+")
plt.xlabel("correlation length")
plt.ylabel("error")
plt.yscale('log', base=10)
plt.savefig('figure_12.jpg')