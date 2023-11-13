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


L_x = 300
L_y = 300
L = 100     #L_y//2
t = 1
t_J = t/5
Delta_s_Trivial = t/5
Delta_p_A1us = t/5
Delta_s_A1us = t/20
mu = -2*t
n = 12      #number of eigenvalues in sparse diagonalization
phi_external = 0
phi_eq = 0.12*2*np.pi    #0.14*2*np.pi
y = np.arange(1, L_y+1)
y_0 = (L_y-L)//2
y_1 = (L_y+L)//2
y_s = (L_y+1)//2

params = {"L_x":L_x, "L_y":L_y, "t":t, "t_J":t_J,
          "Delta_s_Trivial":Delta_s_Trivial,
          "Delta_p_A1us":Delta_p_A1us,
          "Delta_s_A1us":Delta_s_A1us,
          "mu":mu, "n":n, "phi_external":phi_external,
          "phi_eq":phi_eq, "L":L
          }


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
    
    
#%% Saving
import os
my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_directory = "Data"
my_file = "Double_soliton"+";"+";".join(f"{key}={params[key]}" for key, value in params.items())
np.savez(os.path.join(my_path, my_directory, my_file), params=params,
         eigenvalues_sparse=eigenvalues_sparse,
         eigenvectors_sparse=eigenvectors_sparse, index=index,
         probability_density=probability_density)

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

np.savez("L=100", y=y, psi=probability_density_right)