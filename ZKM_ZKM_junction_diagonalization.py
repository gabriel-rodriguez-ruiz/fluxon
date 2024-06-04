#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:58:56 2024

@author: gabriel
"""

import numpy as np
from superconductor import TrivialSparseSuperconductor, \
                            ZKMSparseSuperconductor                            
from junction import Junction, PeriodicJunction, PeriodicJunctionInXAndY
from phase_functions import phase_soliton_antisoliton, phase_single_soliton
import scipy
from functions import get_components
import matplotlib.pyplot as plt


L_x = 150
L_y = 150
L = 0     #odd
t = 1
t_J = t/2#t#t/2
Delta_0 = 0.4#0.2#0.4
Delta_1 = 0.2
Lambda = 0.5
mu = 2*t#t#2*t
n = 12      #number of eigenvalues in sparse diagonalization
phi_external = 0
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+10)//2

# Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
Phi = phase_single_soliton(phi_external, y, y_0)

S_1 = ZKMSparseSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda)
S_2 = ZKMSparseSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda)

# J = PeriodicJunction(S_1, S_2, t_J, Phi)
# J = Junction(S_1, S_2, t_J, Phi)
J = PeriodicJunctionInXAndY(S_1, S_2, t_J, Phi)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(J.matrix, k=n, sigma=0) 

#%% Probability density
index = np.arange(n)   #which zero mode (less than k)
probability_density = []
states = []
for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], J.L_x, J.L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    states.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=-1))
index = 0
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density")
ax.text(0,0, rf'$index={index}$')
plt.tight_layout()

probability_density_right = probability_density[index][:, S_1.L_x-1]/np.linalg.norm(probability_density[index][:, S_1.L_x-1])  #The y-axis is inverted

fig, ax = plt.subplots()
ax.plot(y, probability_density_right, "o")
#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("Probability density at the junction")
ax.text(5,25, rf'$index={index}$')

# np.savez("L=100", y=y, psi=probability_density_right)