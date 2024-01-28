#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:00:35 2023

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def energy_density(x, t):
    return ( 1-np.cos(4*np.arctan(0.1*np.cosh(1.00504*x)*
                                 1/np.sinh(0.100504*t)))+
            (1/np.sinh(0.100504*t)**2*(8.08081*np.cosh(1.00504*x)**2*1/np.tanh(0.100504*t)**2+808.081*np.sinh(1.00504*x)**2))/
            (100+np.cosh(1.00504*x)**2/np.sinh(0.100504*t)**2)**2 )

x = np.linspace(-5, 5, 1000)
# t = np.linspace(-3, 0.000001, 5)
t = np.linspace(-0.00000001, -3, 1000)

X, Y = np.meshgrid(x, t)
plt.style.use('./Images/paper.mplstyle')

fig, ax = plt.subplots(figsize=(3.375, 3.375), subplot_kw={'projection': '3d'})
# ax.plot_wireframe(X, Y, energy_density(X, Y), rcount=20, ccount=0)
ax.plot_wireframe(X, Y, energy_density(X, Y), rcount=3, ccount=0, color=["blue", "green", "orange", "red"])

x = np.linspace(-5, 5, 1000)
# t = np.linspace(-3, 0.000001, 5)
t = np.linspace(3, 0.00000001, 1000)
X, Y = np.meshgrid(x, t)

ax.plot_wireframe(X, Y, energy_density(X, Y), rcount=3, ccount=0, color=["red", "orange", "green", "blue"])

ax.set_xlabel(r"$y$")
ax.set_ylabel(r"$t$")
ax.set_zlabel(r"$e(y,t)$")
ax.set_xlim([-5, 5])
ax.set_ylim([-3, 3])

ax.plot([-5.2, 0], [0, 0], [8, 8], "--")
ax.plot([-5.2, -1.9], [-3, -3], [4, 4], "r--")

# ax.plot(X[265, 265], -5, 0, "o")
# ax.plot(X[734, 734], -5, 0, "o")

# ax.plot(X[np.where(energy_density(X,-5)==np.max(energy_density(X, -5)))[1][0],
#           np.where(energy_density(X,-5)==np.max(energy_density(X, -5)))[1][0]],
#         -5, 0, "o")
# ax.plot(X[np.where(energy_density(X,-4)==np.max(energy_density(X, -4)))[1][0],
#           np.where(energy_density(X,-4)==np.max(energy_density(X, -4)))[1][0]],
#         -4, 0, "o")

for u in np.linspace(-3, 3, 100):
    index = np.where(energy_density(X, u)==np.max(energy_density(X, u)))
    ax.plot( X[index[1][0],
              index[1][0]],
            u, 0, "ok")
    ax.plot(X[index[1][1],
              index[1][1]],
            u, 0, "ok")
plt.tight_layout()

V = 0.1
delta = 1/np.sqrt(1-V**2)
def L(t):
    return 1/delta*np.arccosh(1/V*np.sinh(-delta*V*np.abs(t)*np.tan(-np.pi/4)))

fig, ax = plt.subplots()
t = np.linspace(-3, 3, 1000)
ax.plot(t, [np.exp(-2*L(t_0)) for t_0 in t ])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$E_{+}(t)$")
plt.tight_layout()