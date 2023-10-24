# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:03:39 2023

@author: gabri
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from superconductor import TrivialSuperconductor,\
                            TrivialSuperconductorPeriodicInY,\
                            TrivialSuperconductorKY,\
                            TrivialSparseSuperconductor,\
                            TrivialSparseSuperconductorPeriodicInY,\
                            TrivialSparseSuperconductorKY

L_x = 20
L_y = 20
t = 1
Delta_s = t/2
mu = -2*t
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta_s": Delta_s}
k_values = np.linspace(0, np.pi, L_y)

S = TrivialSuperconductor(L_x, L_y, t, mu, Delta_s)
S_periodic = TrivialSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_s)
S_sparse = TrivialSparseSuperconductor(L_x, L_y, t, mu, Delta_s)
S_sparse_periodic = TrivialSparseSuperconductorPeriodicInY(L_x, L_y, t, mu,
                                                           Delta_s)

eigenvalues_S = np.linalg.eigvalsh(S.matrix) 
eigenvalues_periodic = np.linalg.eigvalsh(S_periodic.matrix)
eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(
                                                S_sparse.matrix, k=6, sigma=0)
eigenvalues_sparse_periodic, eigenvectors_sparse_periodic =\
                                    scipy.sparse.linalg.eigsh(S_sparse.matrix,
                                                        k=6, sigma=0)

fig, ax = plt.subplots()
ax.plot(eigenvalues_periodic, "o", label="Closed")
ax.plot(eigenvalues_S, "o", label="Open")
ax.plot(eigenvalues_sparse_periodic, "o", label="Closed sparse")
ax.plot(eigenvalues_sparse, "o", label="Open sparse")
ax.set_ylabel("E")
ax.set_xlabel("Index of eigenvalues")
plt.title("Eigenvalues in an open or closed system")
plt.legend()

fig, ax = plt.subplots()
eigenvalues_k = []
eigenvalues_sparse_k = []
for k_value in k_values:
    S_KY = TrivialSuperconductorKY(k_value, L_x, t, mu, Delta_s)
    S_sparse_KY = TrivialSparseSuperconductorKY(k_value, L_x, t, mu, Delta_s)
    eigenvalues_k.append(np.linalg.eigvalsh(S_KY.matrix))
    eigenvalues_sparse_k.append(scipy.sparse.linalg.eigsh(S_sparse_KY.matrix,
                                                  k=8, sigma=0)[0])
ax.plot(k_values, eigenvalues_k)
ax.plot(k_values, eigenvalues_sparse_k, "o")

ax.set_ylabel("E")
ax.set_xlabel("k")

Eigenvalues_matrix = np.array(eigenvalues_k)
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
ax.plot(eigenvalues_sparse_periodic, "o", label="Periodic sparse")

ax.set_ylabel("E")
ax.set_xlabel("Index of eigenvalues")
plt.legend()