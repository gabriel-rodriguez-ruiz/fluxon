# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 08:46:18 2023

@author: gabri
"""
import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y
from hamiltonian import Hamiltonian

class TrivialSuperconductor(Hamiltonian):
    r"""
    Trivial superconductor with local s-wave pairing symmetry.
    
    Parameters
    ----------
    L_x : int
        Number of sites in x-direction (horizontal).
    L_y : int
        Number of sites in y-direction (vertical).
    t : float
        Hopping amplitude in x and y directions. Positive.
    mu : float
        Chemical potential.
    Delta_s : float
        Local s-wave pairing potential.
        
    .. math ::
       \vec{c_{n,m}} = (c_{n,m,\uparrow},
                        c_{n,m,\downarrow},
                        c^\dagger_{n,m,\downarrow},
                        -c^\dagger_{n,m,\uparrow})^T
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} (-\mu \vec{c}^\dagger_{n,m}
          \tau_z\sigma_0  \vec{c}_{n,m}) +
           \frac{1}{2} \sum_n^{L_x-1} \sum_m^{L_y} \left(
               \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_x \right] \vec{c}_{n+1,m} 
            + H.c. \right) +
           \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y-1}
           \left( \vec{c}^\dagger_{n,m}\left[ 
            -t\tau_z\sigma_0 -
            i\frac{\Delta}{2} \tau_x\sigma_y \right] \vec{c}_{n,m+1}
            + H.c. \right) 
    """
    def __init__(self, L_x:int, L_y:int, t:float, mu:float, Delta_s:float):
        self.t = t
        self.mu = mu
        self.Delta_s = Delta_s
        onsite = 1/2*(-mu)*np.kron(tau_z, sigma_0)
        hopping_x = 1/2*(-t*np.kron(tau_z, sigma_0) \
                              - 1j/2*Delta_s*np.kron(tau_x, sigma_x))
        hopping_y = 1/2*(-t*np.kron(tau_z, sigma_0) \
                              - 1j/2*Delta_s*np.kron(tau_x, sigma_y))
        super().__init__(L_x, L_y, onsite, hopping_x, hopping_y)


    