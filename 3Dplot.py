from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1,projection='3d')
data=np.load('F:/script/class2vec/real_star_file/10340_case2_E8_ig/umap_3D_normal.npy')
print(data[1])
scale = 8
# Make data.
X = data[:,0]
Y = data[:,1]
Z = data[:,2]

# Plot the surface.
ax.scatter(data[:,0], data[:,1], data[:,2], c='blue', s=10)

# Customize the z axis.
#ax.set_zlim(0, 100)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# rotate the axes and update
for angle in range(0, 360):
   ax.view_init(30, 40)


plt.show()