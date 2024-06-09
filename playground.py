from sklearn.neighbors import KernelDensity
import numpy as np

A = 1

print(np.shape(A))

# # Example data: 100 samples with 2 features
# data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100)

# # Instantiate and fit the KernelDensity model
# kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data)

# # Generate a grid of points where we want to evaluate the density
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(x, y)
# xy = np.vstack([X.ravel(), Y.ravel()]).T

# # Evaluate the density model on the grid
# Z = np.exp(kde.score_samples(xy))  # score_samples returns the log of the density

# # Reshape back to a 2D grid for plotting
# Z = Z.reshape(X.shape)

# # Plotting
# import matplotlib.pyplot as plt

# plt.contourf(X, Y, Z, levels=50, cmap='viridis')
# plt.colorbar()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Multivariate Kernel Density Estimation')
# plt.show()
