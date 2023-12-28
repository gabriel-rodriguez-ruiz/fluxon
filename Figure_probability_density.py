#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:25:26 2023

@author: gabriel
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import root
def trascendental_equation(k, m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    A function whose roots represents the trascendental equation.
        (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)=0
    """
    return (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)

def Kappa(m_0, Delta, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    (y, kappa, m_0, Delta, L)m_0 : float
        Mass.
    Delta : floar
        Gap.
    L : float
        Length.

    Returns
    -------
    The wavevector k solving:
        (m_0/Delta)**2 - k**2 = (m_0/Delta)**2 * np.exp(-2*k*L)
    """
    return root(trascendental_equation, 1, args=(m_0, Delta, L)).x

def density(x):
    if x<=0:
        return kappa**2*np.exp(2*kappa*(x-L))
    elif 0<x<L:
        return (-(kappa+1)*np.exp(kappa*(x-L)) + np.exp(-kappa*(x+L)))**2
    else:
        return (-(kappa+1)*np.exp(kappa*L) + np.exp(-kappa*L))**2 * np.exp(-2*kappa*x)

plt.style.use('./Images/paper.mplstyle')


L = 3
x = np.linspace(-5, L+5, 1000)
kappa = float(Kappa(1,  1, L))

fig, ax = plt.subplots(2)
ax[0].plot(x, np.exp(-2*kappa*np.abs(x)))
ax[0].set_xticks([0], [r"$0$"])
ax[0].set_yticks([1], [r"$|C|^2$"])
ax[0].set_xlabel("y")
ax[0].arrow(-1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), 1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
ax[0].arrow(1/(2*kappa), np.exp(-2*kappa*np.abs(-1/(2*kappa))), -1/kappa, 0, color="k", width= 0.01, length_includes_head=True)
ax[0].text(-0.3, np.exp(-2*kappa*np.abs(-1/(2*kappa)))-0.2, r"$1/\kappa$", fontsize=8)
ax[0].text(-5, 0.8, "(a)", fontsize=10)

ax[1].plot(x, [density(x_0) + density(-x_0 + L) for x_0 in x])
ax[1].set_xticks([0, L], [r"$0$", "L"])
ax[1].set_yticks([density(L)], [""])
ax[1].set_xlabel(r"$y$")
ax[1].text(-5, 3.2, "(b)", fontsize=10)

plt.tight_layout()

plt.subplots_adjust(hspace=0.4)
plt.savefig("./Images/Probability_density.pdf")

