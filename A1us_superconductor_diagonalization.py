# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 07:40:10 2023

@author: gabri
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from superconductor import A1usSparseSuperconductor,\
                            A1usSparseSuperconductorPeriodicInY,\
                            A1usSuperconductorKY
from functions import get_components

L_x = 200
L_y = 100
t = 1
Delta_s = t/20
Delta_p = t/40#t/5
mu = -2*t
n = 12      #number of eigenvalues in sparse diagonalization
k_values = np.linspace(0, np.pi, L_y)
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta_s": Delta_s}

# S = A1usSparseSuperconductor(L_x, L_y, t, mu, Delta_s, Delta_p)
S = A1usSparseSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_s, Delta_p)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(S.matrix, k=n, sigma=0) 

#%% Spectrum vs. k

fig, ax = plt.subplots()
eigenvalues_k = []
eigenvectors_k = []
for k_value in k_values:
    S_KY = A1usSuperconductorKY(k_value, L_x, t, mu, Delta_s, Delta_p)
    eigenvalues, eigenvectors = np.linalg.eigh(S_KY.matrix)
    eigenvalues_k.append(eigenvalues)
    eigenvectors_k.append(eigenvectors)

ax.plot(k_values, eigenvalues_k)
ax.set_ylabel("E")
ax.set_xlabel("k")

#%% Probability density
index = np.arange(n)   #which zero mode (less than k)
probability_density = []
for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], L_x, L_y)
    # probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    probability_density.append(np.abs(destruction_up))
index = 0
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density")
ax.text(0,0, rf'$index={index}$')
plt.tight_layout()

#%%
import os
my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_file = 'graph.png'
#fig.savefig(os.path.join(my_path, my_file))     