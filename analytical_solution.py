#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:44:08 2023

@author: gabriel
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


def trascendental_equation(k, m_0, v, L):
    """
    Wavevector of the localized state.

    Parameters
    ----------
    m_0 : float
        Mass.
    v : float
        Velocity.
    L : float
        Length.

    Returns
    -------
    A function whose roots represents the trascendental equation.
        (m_0/Delta)**2 - k**2 - (m_0/Delta)**2 * np.exp(-2*k*L)=0
    """
    return (m_0/v)**2 - k**2 - (m_0/v)**2 * np.exp(-2*k*L)

def Kappa(m_0, v, L):
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
    return root(trascendental_equation, m_0/v, args=(m_0, v, L)).x

def psi_1_prime(y, kappa, m_0, v, L):
    alpha = kappa*v/m_0
    C = np.sqrt(kappa/(2*(alpha*(alpha+1)-np.exp(-2*kappa*L)*(alpha+(alpha+1)*kappa*L))))
    if y<=0:
        return -C*alpha*np.exp(kappa*(y-L))
    elif (y>0 and y<=L):
        return C*(-(alpha+1)*np.exp(kappa*(y-L)) + np.exp(-kappa*(y+L)) )
    else:
        return C*(-(alpha+1)*np.exp(kappa*L) + np.exp(-kappa*L)) * np.exp(-kappa*y)

# def psi_2_prime_plus(y, kappa, m_0, v, L):
#     return 1j*psi_1_prime(-y+L, kappa, m_0, v, L)
# def psi_2_prime_minus(y, kappa, m_0, v, L):
#     return -1j*psi_1_prime(-y+L, kappa, m_0, v, L)

def psi_1_plus(y, kappa, m_0, v, L):
    return 1/2*(psi_1_prime(y, kappa, m_0, v, L) -1j*psi_1_prime(-y+L, kappa, m_0, v, L))
# def psi_1_minus(y, kappa, m_0, v, L):
#     return 1/2*(psi_1_prime(y, kappa, m_0, v, L) + psi_2_prime_minus(y, kappa, m_0, v, L))
# def psi_2_plus(y, kappa, m_0, v, L):
#     return 1/2*(psi_1_prime(y, kappa, m_0, v, L) +1j* psi_1_prime(y, kappa, m_0, v, L))
# def psi_2_minus(y, kappa, m_0, Delta, L):
#     return 1/2*(psi_1_prime(y, kappa, m_0, Delta, L) -1j* psi_2_prime_minus(y, kappa, m_0, Delta, L))
# def psi_3_plus(y, kappa, m_0, Delta, L):
#     return 1/2*(-psi_1_prime(y, kappa, m_0, Delta, L) + 1j*psi_2_prime_plus(y, kappa, m_0, Delta, L))
# def psi_3_minus(y, kappa, m_0, Delta, L):
#     return 1/2*(-psi_1_prime(y, kappa, m_0, Delta, L) + 1j*psi_2_prime_minus(y, kappa, m_0, Delta, L))
# def psi_4_plus(y, kappa, m_0, Delta, L):
#     return 1/2*(1j*psi_1_prime(y, kappa, m_0, Delta, L) - psi_2_prime_plus(y, kappa, m_0, Delta, L))
# def psi_4_minus(y, kappa, m_0, Delta, L):
#     return 1/2*(1j*psi_1_prime(y, kappa, m_0, Delta, L) - psi_2_prime_minus(y, kappa, m_0, Delta, L))

# def psi_plus(y, kappa, m_0, Delta, L):
#     return 1/2*np.array([psi_1_plus(y, kappa, m_0, Delta, L), psi_2_plus(y, kappa, m_0, Delta, L), psi_3_plus(y, kappa, m_0, Delta, L), psi_4_plus(y, kappa, m_0, Delta, L)])

