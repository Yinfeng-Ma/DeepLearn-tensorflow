#coding:UTF-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import hermite
x=np.arange(-5,5,0.05)
n=len(x)
dx=x[1]-x[0]

h=np.zeros((n,n),dtype=np.double)

for i in range(n-1):
    h[i][i]=0.5*x[i]**2+1/dx**2
    h[i][i+1]=-0.5/dx**2
    h[i+1][i]=-0.5/dx**2
h[n-1][n-1]=0.5*x[n-1]**2+1/dx**2
a,b=np.linalg.eig(h)

sort_index=np.argsort(a)
psi=b[:,sort_index]

psi0=psi[:,0]

norm1=np.trapz(psi**2,x,axis=0)
norm2=simps(psi**2,x,axis=0)

y=np.exp(-0.5*x**2)/np.pi**0.25
y1=np.exp(-0.5*x**2)*hermite(1)(x)/np.sqrt(2)/np.pi**0.25


norm3=simps(y1**2,x)
print norm3
psi=psi/np.sqrt(norm1)

plt.subplot(121)

plt.plot(x,-psi[:,0],'r',x,y,'--k')

plt.xlabel('x')
plt.ylabel(r'$\psi$')
plt.legend([r'$\psi$',r'$\psi_0^{har}$'])
plt.subplot(122)
plt.plot(x,-psi[:,1],x,y1,'--k')

plt.xlabel('x')
plt.ylabel(r'$\psi$')
plt.legend([r'$\psi$',r'$\psi_1^{har}$'])
plt.show()
