#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:14:13 2023

@author: gabriel
"""
import numpy as np
from superconductor import TrivialSparseSuperconductor, \
                            A1usSparseSuperconductor                            
from junction import Junction, PeriodicJunction
from phase_functions import phase_soliton_antisoliton_S_around_zero
import scipy
from functions import get_components
import matplotlib.pyplot as plt


L_x = 100
L_y = 500
L = 240     #L_y//2
t = 1
t_J = t/10
Delta_s_Trivial = t/2
Delta_p_A1us = t/2
Delta_s_A1us = t/10
mu = -2*t
n = 12      #number of eigenvalues in sparse diagonalization
phi_external = 0
phi_eq = 0.05*2*np.pi    #0.14*2*np.pi
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2

Phi = phase_soliton_antisoliton_S_around_zero(phi_external, phi_eq, y, y_0, y_1)

S_A1us = A1usSparseSuperconductor(L_x, L_y, t, mu, Delta_s_A1us, Delta_p_A1us)
S_Trivial = TrivialSparseSuperconductor(L_x, L_y, t, mu, Delta_s_Trivial)

# J = Junction(S_A1us, S_Trivial, t_J, Phi)
J = PeriodicJunction(S_A1us, S_Trivial, t_J, Phi)

eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(J.matrix, k=n, sigma=0) 

#%% Probability density
index = np.arange(n)   #which zero mode (less than k)
probability_density = []
for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], J.L_x, J.L_y)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    
index = 2
fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density")
ax.text(0,0, rf'$index={index}$')
plt.tight_layout()

probability_density_right = probability_density[index][:, S_A1us.L_x-1]/np.linalg.norm(probability_density[index][:, S_A1us.L_x-1])  #The y-axis is inverted

fig, ax = plt.subplots()
ax.plot(y, probability_density_right, "o")
#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel(r"$\ell$")
ax.set_ylabel("Probability density at the junction")
ax.text(5,25, rf'$index={index}$')
