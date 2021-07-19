from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

n = 256

rvalues = pd.read_csv('r_values.csv')
zvalues = pd.read_csv('z_values.csv')

rmean = 0.0
zmean = np.mean(zvalues['z_values'])
rradius = np.max(rvalues['r_values'])
zradius = np.max(zvalues['z_values'])
plot_radius = np.max([rradius,zradius])

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122,projection='3d')
y = zvalues['z_values']
x = rvalues['r_values']
t = np.linspace(0, np.pi*2, n)

xn = np.outer(x, np.cos(t))
yn = np.outer(x, np.sin(t))
zn = np.zeros_like(xn)

for i in range(len(x)):
    zn[i:i+1,:] = np.full_like(zn[0,:], y[i])

ax1.scatter(x, y)
ax2.plot_surface(xn, yn, zn)
ax2.set_xlim3d([ -1.0* plot_radius, plot_radius])
ax2.set_ylim3d([-1.0* plot_radius, plot_radius])
ax2.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
#plt.gca().set_aspect('equal', adjustable='box')
plt.show()