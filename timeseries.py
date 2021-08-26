import matplotlib.pyplot as plt
from matplotlib import animation
from numpy import random 
import pandas as pd

fig = plt.figure()
ax1 = plt.axes(xlim=(-1, 1), ylim=(-0.1,1.2))
line, = ax1.plot([], [], lw=2)
plt.xlabel('r')
plt.ylabel('z')

linecolor = "blue"
ax1.axvline(0, linewidth = 0.8)
plotlays, plotcols = [2], [linecolor,linecolor]
lines = []
ax1.set_aspect('equal','box')
for index in range(2):
    lobj = ax1.plot([],[],lw=2,color=plotcols[index])[0]
    lines.append(lobj)


def init():
    for line in lines:
        line.set_data([],[])
    return lines

x1,y1 = [],[]
x2,y2 = [],[]


def integers(a, b):
    return list(range(a, b+1))

def animate(i):
    rvalues = pd.read_csv('build/solutions/r_values_' + str(i) + '.csv')
    zvalues = pd.read_csv('build/solutions/z_values_' + str(i) + '.csv')

    x1 = rvalues['r_values']
    y1 = zvalues['z_values']

    x2 = -1.0*x1
    y2 = y1

    xlist = [x1, x2]
    ylist = [y1, y2]

    #for index in range(0,1):
    for lnum,line in enumerate(lines):
        line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=integers(1,2000), interval=5, blit=True)



#anim.save('animation.mp4', fps=80, dpi = 200)
plt.show()


