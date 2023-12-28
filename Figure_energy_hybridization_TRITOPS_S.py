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
# my_file = "TRITOPS_S_spectrum_vs_distance_between_soliton;L_min=10;L_max=100;L_x=300;L_y=300;t=1;t_J=0.2;Delta_s_Trivial=0.2;Delta_p_A1us=0.2;Delta_s_A1us=0.05;mu=-2;n=12;phi_external=0.0;phi_eq=0.754.npz"
my_file = "TRITOPS_S_spectrum_vs_distance_between_soliton;L_min=10;L_max=100;L_x=300;L_y=300;t=1;t_J=0.2;Delta_s_Trivial=0.2;Delta_p_A1us=0.2;Delta_s_A1us=0.05;mu=-2;n=36;phi_external=0.0;phi_eq=0.754.npz"


data = np.load(os.path.join(my_path, my_directory, my_file),
               allow_pickle=True)

#data.files
params = data["params"].tolist()
E_numerical = data["E_numerical"]
L_values = data["L_values"]
n = params["n"]

E_numerics = E_numerical[n//2+1]
plt.style.use('./Images/paper.mplstyle')

fig, ax = plt.subplots(dpi=300)
ax.plot(L_values, E_numerical[n//2+1], ".")
ax.plot(L_values, E_numerical[n//2+2], ".")
ax.plot(L_values, E_numerical[n//2+3], ".")
ax.plot(L_values, E_numerical[n//2+4], ".")
ax.plot(L_values, E_numerical[n//2+5], ".")
ax.plot(L_values, E_numerical[n//2+6], ".")
ax.plot(L_values, E_numerical[n//2+7], ".")
ax.plot(L_values, E_numerical[n//2+8], ".")
ax.plot(L_values, E_numerical[n//2+9], ".")
ax.plot(L_values, E_numerical[n//2+10], ".")
ax.plot(L_values, E_numerical[n//2+11], ".")
ax.plot(L_values, E_numerical[n//2+12], ".")
ax.plot(L_values, E_numerical[n//2+13], ".")
ax.plot(L_values, E_numerical[n//2+14], ".")
ax.plot(L_values, E_numerical[n//2+15], ".")

E_numerics = [E_numerical[n//2+1][0],
              E_numerical[n//2+1][1],
              E_numerical[n//2+15][2],
              E_numerical[n//2+9][3],
              E_numerical[n//2+5][4],
              E_numerical[n//2+3][5],
              E_numerical[n//2+3][6],
              E_numerical[n//2+1][7],
              E_numerical[n//2+1][8],
              E_numerical[n//2+1][9]]

ax.plot(L_values, E_numerics, "o")
ax.set_yscale('log')

ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()