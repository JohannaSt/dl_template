import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

'''
def f(x, y):
    if x<0.5:
        print x
    return (0.0+0.15*(((x-127/2)/(127/2))**2)+0.15*(((y-127/2)/(127/2))**2))
    #return (0.15*abs(((x-127/2)/(127/2)))+0.15*abs(((y-127/2)/(127/2))))
'''
X=np.linspace(0,127,127)
Y=np.linspace(0,127,127)
#x, y = np.meshgrid(X, Y)
#Z = f(x,y)

Z = np.zeros((len(X), len(Y)))

for i in range(len(X)):
    for j in range(len(Y)):
        if np.sqrt(abs ((X[i]-127/2)/(127/2))**2+abs ((Y[j]-127/2)/(127/2))**2)<0.5:
        #if abs ((X[i]-127/2)/(127/2)) < 0.5 and abs ((Y[j]-127/2)/(127/2)) < 0.5:
            Z[i, j] = (((X[i]-127/2)/(127/2))**2)+(((Y[j]-127/2)/(127/2))**2)
        else:
            Z[i, j] = 0.3

def g(n):
    return ((n-127/2)/(127/2))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(g(X), g(Y), Z, 50, cmap='binary')

plt.show()
