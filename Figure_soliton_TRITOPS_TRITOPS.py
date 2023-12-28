#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:48:21 2023

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np
from phase_functions import phase_single_soliton_arctan, phase_single_soliton_S, phase_soliton_antisoliton_arctan

y = np.linspace(-100, 100, 1000)
y_0 = 0
lambda_J = 10
phi_external = 0
phi_0 = 0.12*2*np.pi


Phi = phase_single_soliton_arctan(phi_external, y, y_0, lambda_J)

plt.style.use('./Images/paper.mplstyle')

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.375,3.375))
ax.plot(y, Phi/(2*np.pi), "b", label=r"$\phi_1/(2\pi)$")
ax.set_ylabel(r"$\phi/(2\pi)$")
ax2.set_ylabel(r"$m(\phi)/m_0$")
ax2.plot(y, np.cos(Phi/2), "b", label=r"$m(\phi_1)/m_0$")
ax.set_xticks([0], [r"$y_0$"])
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1], minor=True)
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)

ax.text(-100, 0.75, "(a)")
ax2.text(-100, 0.5, "(b)")


plt.savefig("./Images/Phase_TRITOPS_TRITOPS.pdf")

