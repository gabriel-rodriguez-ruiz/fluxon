# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:03:39 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
from superconductor import TrivialSuperconductor,\
                            TrivialSuperconductorPeriodicInY,\
                            TrivialSuperconductorKY

L_x = 20
L_y = 20
t = 1
Delta_s = t/2
mu = -2*t
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta_s": Delta_s}
k_values = np.linspace(0, np.pi, L_y)

S = TrivialSuperconductor(L_x, L_y, t, mu, Delta_s)
S_periodic = TrivialSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_s)
# S = TrivialSuperconductorKY(k, L_x, t, mu, Delta_s)

eigenvalues_S = np.linalg.eigvalsh(S.matrix) 
eigenvalues_periodic = np.linalg.eigvalsh(S_periodic.matrix) 
fig, ax = plt.subplots()
ax.plot(eigenvalues_periodic, "o", label="Closed")
ax.plot(eigenvalues_S, "o", label="Open")

ax.set_ylabel("E")
ax.set_xlabel("Index of eigenvalues")
plt.title("Eigenvalues in an open or closed system")
plt.legend()

fig, ax = plt.subplots()
eigenvalues = []
for k_value in k_values:
    S_KY = TrivialSuperconductorKY(k_value, L_x, t, mu, Delta_s)
    eigenvalues.append(np.linalg.eigvalsh(S_KY.matrix)) 

ax.plot(k_values, eigenvalues, "o")
ax.set_ylabel("E")
ax.set_xlabel("k")

Eigenvalues_matrix = np.array(eigenvalues)
eigenvalues_vector = Eigenvalues_matrix.flatten()
fig, ax = plt.subplots()
ax.plot(np.sort(eigenvalues_vector), "o")
ax.set_ylabel("E")
ax.set_xlabel("Index of eigenvalues")
plt.title("All k-eigenvalues")

#%% All together
fig, ax = plt.subplots()
ax.plot(eigenvalues_periodic, "o", label="Periodic")
ax.plot(np.sort(eigenvalues_vector), "o", label="All k_eigenvalues")

ax.set_ylabel("E")
ax.set_xlabel("Index of eigenvalues")
plt.legend()