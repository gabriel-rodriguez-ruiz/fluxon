#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:37:20 2023

@author: gabriel
"""
import numpy as np
from superconductor import A1usSuperconductorKY, \
                            TrivialSuperconductorKY
from functions import phi_spectrum       
from junction import Junction, PeriodicJunction
import scipy
import matplotlib.pyplot as plt

L_x = 200
t = 1
Delta_s_Trivial = t/2
Delta_p_A1us = t/2      #topologic if Delta_p>Delta_s
Delta_s_A1us = t/4
mu = -2*t
t_J = t/10
phi_values = np.linspace(0, 2*np.pi, 20)
k_values = np.linspace(0, np.pi, 20)

eigenvalues = []
for k in k_values:
    eigenvalues_k = []
    print(k)
    for phi in phi_values:
        phi = np.array([phi])   #array of length 1
        S_A1us = A1usSuperconductorKY(k, L_x, t, mu, Delta_s_A1us, Delta_p_A1us)
        S_Trivial = TrivialSuperconductorKY(k, L_x, t, mu, Delta_s_Trivial)
        J = Junction(S_A1us, S_Trivial, t_J, phi)
        energies = np.linalg.eigvalsh(J.matrix.toarray())
        energies = list(energies)
        eigenvalues_k.append(energies)
    eigenvalues.append(eigenvalues_k)
eigenvalues = np.array(eigenvalues)
E_phi = eigenvalues

#%% Plotting for a given k

fig, ax = plt.subplots()
j = 5   #index of k-value
for i in range(np.shape(E_phi)[2]):
    plt.plot(phi_values, E_phi[j, :, i], ".k", markersize=1)

plt.title(f"k={k_values[j]}")
plt.xlabel(r"$\phi$")
plt.ylabel(r"$E_k$")

#%% Total energy

E_positive = E_phi[:, :, np.shape(E_phi)[2]//2:]
total_energy_k = np.sum(E_positive, axis=2)
total_energy = np.sum(total_energy_k, axis=0) 
phi_eq = phi_values[np.where(min(-total_energy)==-total_energy)]

#%% Josephson current

Josephson_current = np.diff(-total_energy)
Josephson_current_k = np.diff(-total_energy_k)

J_0 = np.max(Josephson_current) 
fig, ax = plt.subplots()
ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
ax.set_title("Josephson current for given k")

for i, k in enumerate(k_values):
    ax.plot(phi_values[:-1]/(2*np.pi), Josephson_current_k[i,:])

#%% Plotting of total energy

plt.rc("font", family="serif")  # set font family
plt.rc("xtick", labelsize="large")  # reduced tick label size
plt.rc("ytick", labelsize="large")
plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc("text", usetex=True) # for better LaTex (slower)
plt.rcParams['xtick.top'] = True    #ticks on top
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['ytick.right'] = True    #ticks on left
plt.rcParams['ytick.labelright'] = False
plt.rc('legend', fontsize=18) #fontsize of the legend

def energy(phi_values, E_0, E_J):
    # return E_0*(4*np.cos(phi_eq[0])*(1-np.cos(phi_values))-2*np.sin(phi_values)**2) 
    return E_J*(1-np.cos(phi_values)) - 2*E_0*np.sin(phi_values)**2
popt, pcov = scipy.optimize.curve_fit(energy, xdata = phi_values, ydata = -total_energy+total_energy[0])
E_0 = popt[0]
E_J = popt[1]
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), -total_energy+total_energy[0], label="Numerical")
ax.plot(phi_values/(2*np.pi), energy(phi_values, E_0, E_J))
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$E(\phi)$")
ax.set_title(r"$\phi_{0}=$"+f"{(2*np.pi-phi_eq[0])/(2*np.pi):.2}"+r"$\times 2\pi$")

# plt.legend()
plt.tight_layout()