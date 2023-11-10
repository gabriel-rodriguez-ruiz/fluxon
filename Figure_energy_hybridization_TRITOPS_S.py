#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:27:07 2023

@author: gabriel
"""
import matplotlib.pyplot as plt
import numpy as np
import os

my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_directory = "Data"
my_file = "TRITOPS_S_spectrum_vs_distance_between_soliton;L_min=10;L_max=100;L_x=300;L_y=300;t=1;t_J=0.2;Delta_s_Trivial=0.2;Delta_p_A1us=0.2;Delta_s_A1us=0.05;mu=-2;n=12;phi_external=0.0;phi_eq=0.754.npz"
data = np.load(os.path.join(my_path, my_directory, my_file),
               allow_pickle=True)

#data.files
params = data["params"].tolist()
E_numerical = data["E_numerical"]
L_values = data["L_values"]
n = params["n"]

E_numerics = E_numerical[n//2+1]
# plt.style.use('./Images/paper.mplstyle')

fig, ax = plt.subplots(dpi=300)
ax.plot(L_values, abs(E_numerical[0]), "o")
ax.plot(L_values, E_numerical[n//2+1], "*")
ax.plot(L_values, E_numerical[n//2+2], ".")
ax.plot(L_values, E_numerical[n//2+3], ".")
ax.plot(L_values, E_numerical[n//2+4], ".")
ax.plot(L_values, E_numerical[n//2+5], ".")

# ax.plot(L_values, E_numerics, "o")
ax.set_yscale('log')

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()