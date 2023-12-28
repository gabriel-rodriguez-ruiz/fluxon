#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 15:05:37 2023

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_directory = "Data"
my_file = "phi_eq=0.13.npz"

data = np.load(os.path.join(my_path, my_directory, my_file),
               allow_pickle=True)

E = data["E"]
phi_values = data["phi"]
phi_eq = data["phi_eq"]

plt.style.use('./Images/paper.mplstyle')

def energy(phi_values, E_0, E_J):
    # return E_0*(4*np.cos(phi_eq[0])*(1-np.cos(phi_values))-2*np.sin(phi_values)**2) 
    return E_J*(1-np.cos(phi_values)) - 2*E_0*np.sin(phi_values)**2
popt, pcov = scipy.optimize.curve_fit(energy, xdata = phi_values, ydata = E)
E_0 = popt[0]
E_J = popt[1]
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), E, label="Numerical")
ax.plot(np.linspace(0, 1, 1000), energy(np.linspace(0, 2*np.pi, 1000), E_0, E_J), linestyle="dashed")
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$E(\phi)$")
# ax.set_title(r"$\phi_{0}=$"+f"{(2*np.pi-phi_eq[0])/(2*np.pi):.2}"+r"$\times 2\pi$")

plt.tight_layout()
plt.savefig("./Images/Josephson_Energy.pdf")