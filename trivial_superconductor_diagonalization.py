# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:03:39 2023

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt
from superconductor import TrivialSuperconductor,\
                            TrivialSuperconductorPeriodicInY

L_x = 20
L_y = 20
t = 1
Delta_s = t/2
mu = -2*t
params = {"t": t, "mu": mu, "L_x": L_x, "L_y": L_y, "Delta_s": Delta_s}

# S = TrivialSuperconductor(L_x, L_y, t, mu, Delta_s)
S = TrivialSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_s)
eigenvalues, eigenvectors = np.linalg.eigh(S.matrix) 

fig, ax = plt.subplots()
ax.plot(eigenvalues, "o")
ax.set_ylabel("E")
ax.set_xlabel("Indices of eigenvalues")
