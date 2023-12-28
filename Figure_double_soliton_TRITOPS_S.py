#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:55:23 2023

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
import os

my_directory = "/home/gabriel/OneDrive/Doctorado-DESKTOP-JBOMLCA/Archivos/Double_soliton_TRITOPS_S"
my_file = "Double_soliton;L_x=300;L_y=300;t=1;t_J=0.2;Delta_s_Trivial=0.2;Delta_p_A1us=0.2;Delta_s_A1us=0.05;mu=-2;n=12;phi_external=0;phi_eq=0.7539822368615503;L=100.npz"


data = np.load(os.path.join(my_directory, my_file),
               allow_pickle=True)

probability_density = data["probability_density"]
params = data["params"].tolist()

index = 2
plt.style.use('./Images/paper.mplstyle')

fig, ax = plt.subplots()
image = ax.imshow(probability_density[index], cmap="Blues", origin="lower") #I have made the transpose and changed the origin to have xy axes as usually
plt.colorbar(image)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Probability density")
ax.text(0,0, rf'$index={index}$')
plt.tight_layout()

# probability_density_right = probability_density[index][:, params["L_x"]-1]/np.linalg.norm(probability_density[index][:, params["L_x"]-1])  #The y-axis is inverted
probability_density_right = probability_density[index][:, params["L_x"]-1]/np.sum(probability_density[index][:, params["L_x"]-1])  #The y-axis is inverted

y = np.arange(1, params["L_y"]+1)

fig, ax = plt.subplots()
ax.plot(y, probability_density_right, "o")
#ax.plot(np.arange(1, L_y+1), probability_density[index][:, L_x//2-1])
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\rho_{N_x/2,\ell}^{(+)}$")
ax.text(5,25, rf'$index={index}$')

np.savez("L=100", y=y, psi=probability_density_right)

#%%
from analytical_solution import psi_1_plus, Kappa

L = params["L"]
m_0 = 0.0037
v = 0.088       #v
# kappa = m_0/Delta

# kappa = m_0/Delta
kappa = Kappa(m_0, v, L)
z = np.linspace(-100, 200, 1000)

ax.plot(z+101, [2*np.abs(psi_1_plus(z, kappa, m_0, v, L))**2 for z in z])

plt.tight_layout()  
plt.savefig("./Images/Density_at_the_junction.pdf")