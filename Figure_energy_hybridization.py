#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:27:07 2023

@author: gabriel
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

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

fig, ax = plt.subplots()

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

ax.set_yscale('log')

m_numerical, b_numerical = np.polyfit(L_values[2:], np.log(E_numerics[2:]), 1)
x = np.linspace(30, 100)
ax.plot(x, np.exp(b_numerical)*np.exp(m_numerical*x), label=f"{m_numerical:.2}L{b_numerical:.2}")
ax.plot(L_values[2:], E_numerics[2:], "o", label="TRITOPS-S")


ax.set_xlabel(r"$L$")
ax.set_ylabel(r"$E(L)$")

#%% TRITOPS-TRITOPS

my_path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
my_directory = "Data"
# my_file = "TRITOPS_TRITOPS_spectrum_vs_distance_between_soliton;L_min=1;L_max=10;L_x=150;L_y=300;t=1;t_J=0.04;Delta_p_A1us=0.2;Delta_s_A1us=0;mu=-2;n=12;phi_external=0.0;phi_eq=0.npz"
my_file = "TRITOPS_TRITOPS_spectrum_vs_distance_between_soliton;L_min=30;L_max=100;L_x=150;L_y=300;t=1;t_J=0.04;Delta_p_A1us=0.2;Delta_s_A1us=0;mu=-2;n=12;phi_external=0.0;phi_eq=0.npz"

data = np.load(os.path.join(my_path, my_directory, my_file),
               allow_pickle=True)

#data.files
params = data["params"].tolist()
E_numerical = data["E_numerical"]
L_values = data["L_values"]
n = params["n"]

E_numerics = E_numerical[n//2+1]
# ax.plot(L_values, E_numerical[n//2], ".")
# ax.plot(L_values, E_numerical[n//2+1], ".")
# ax.plot(L_values, E_numerical[n//2+2], ".")
# ax.plot(L_values, E_numerical[n//2+3], ".")
# ax.plot(L_values, E_numerical[n//2+4], ".")
# ax.plot(L_values, E_numerical[n//2+5], ".")
m_numerical, b_numerical = np.polyfit(L_values[2:], np.log(E_numerics[2:]), 1)
x = np.linspace(30, 100)
ax.plot(x, np.exp(b_numerical)*np.exp(m_numerical*x), label=f"{m_numerical:.2}L{b_numerical:.2}")

ax.plot(L_values, E_numerics, "s", label="TRITOPS-TRITOPS")

ax.xaxis.set_minor_locator(MultipleLocator(10))
# plt.legend()
plt.tight_layout()

ax.text(50, 1.5e-4, "TRITOPS-TRITOPS", color="g",
        rotation=-35, rotation_mode='anchor',
        fontsize=8)

ax.text(50, 6e-4, "TRITOPS-S", color="b",
        rotation=-24, rotation_mode='anchor',
        fontsize=8)


plt.savefig("./Images/hybridization.pdf")