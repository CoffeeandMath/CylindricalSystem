import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#plt.style.use('seaborn')
plt.style.use('bmh')
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

rvalues = pd.read_csv('r_values.csv')
zvalues = pd.read_csv('z_values.csv')



#ax1.plot(rvalues['S_values'], rvalues['r_values'], label='r Curve',color='black')
#ax1.plot(zvalues['S_values'], zvalues['z_values'], label='z Curve',color='black')
#mag = 0.5
#ax1.plot(rvalues['S_values'], 1.0 - mag*(rvalues['S_values']-0.5)**2.0)
#ax1.set_xlabel('S')
#ax1.set_title('Deformation Curve')
#ax1.legend(frameon=True)

#ax1.invert_xaxis()


#ax1 = fig.gca(projection = '3d')

ax2.plot(zvalues['z_values'],rvalues['r_values'], label='R vs Z Curve',color='black')
ax2.set_xlabel('z')
ax2.set_ylabel('r')
ax2.legend()
ax2.set_ylim([0, 1.5*rvalues['r_values'].max()])









#ax1.tight_layout()

#ax1.savefig('plot.png')
#plt.gca().set_aspect('equal', adjustable='box')
plt.show()