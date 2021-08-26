import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

#plt.style.use('seaborn')
plt.style.use('bmh')
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

rvalues1 = pd.read_csv('build/r_values_0.01.csv')
zvalues1 = pd.read_csv('build/z_values_0.01.csv')
rvalues2 = pd.read_csv('build/r_values_0.05.csv')
zvalues2 = pd.read_csv('build/z_values_0.05.csv')

color1 = 'black'
color2 = 'blue'

ax1.plot(rvalues1['S_values'], rvalues1['r_values'], label='r Curve',color=color1)
ax1.plot(zvalues1['S_values'], zvalues1['z_values'], label='z Curve',color=color1)
ax1.plot(rvalues2['S_values'], rvalues2['r_values'], label='r Curve',color=color2)
ax1.plot(zvalues2['S_values'], zvalues2['z_values'], label='z Curve',color=color2)
mag = 0.5
#ax1.plot(rvalues['S_values'], 1.0 - mag*(rvalues['S_values']-0.5)**2.0)
ax1.set_xlabel('S')
ax1.set_title('Deformation Curve')
ax1.legend(frameon=True)

#ax1.invert_xaxis()

ax2.plot(zvalues1['z_values'],rvalues1['r_values'], label='h = 0.01',color=color1)
ax2.plot(zvalues2['z_values'],rvalues2['r_values'], label='h = 0.05',color=color2)
ax2.set_xlabel('z')
ax2.set_ylabel('r')
ax2.legend()
ax2.set_ylim([0, 1.5*rvalues1['r_values'].max()])









#ax1.tight_layout()

#ax1.savefig('plot.png')

plt.show()