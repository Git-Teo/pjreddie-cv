from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import math

def gaussian_2d(x, y, sigma):
    return math.exp(-(((x**2) + (y**2))/(2*(sigma**2)))) * (1 / (2*math.pi*(sigma**2)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 1.5
ticks = 0.1
sigma = 0.5
xs = np.arange(-n, n+ticks, ticks)
ys = np.arange(-n, n+ticks, ticks)
xs, ys = np.meshgrid(xs, ys)
zs = []
for xr, yr in zip(xs, ys):
    z = [gaussian_2d(x, y, sigma) for x, y in zip(xr, yr)]
    zs.append(z)

ax.set_zlim(0, 0.8)

ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.jet)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()