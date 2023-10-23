# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 08:46:18 2023

@author: gabri
"""
import numpy as np
from pauli_matrices import tau_x, sigma_x, tau_z, sigma_0, sigma_y
from hamiltonian import Hamiltonian, PeriodicHamiltonianInY

class TrivialSuperconductor(Hamiltonian):
    r"""Trivial superconductor with local s-wave pairing symmetry.
    
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
       
       H = \frac{1}{2} \sum_n^{L_x} \sum_m^{L_y} \vec{c}^\dagger_{n,m} \left(-\mu 
          \tau_z\sigma_0 +\Delta_s\tau_x\sigma_0 \right) \vec{c}_{n,m}
       + \frac{1}{2}
       \sum_n^{L_x-1}\sum_m^{L_y}\left[\mathbf{c}^\dagger_{n,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{n+1,m}
       + H.c.\right]
       + \frac{1}{2}
       \sum_n^{L_x}\sum_m^{L_y-1}\left[\mathbf{c}^\dagger_{n,m}\left(-t\tau_z\sigma_0 \right)\mathbf{c}_{n,m+1}
       + H.c.\right]
    """
    def __init__(self, L_x:int, L_y:int, t:float, mu:float, Delta_s:float):
        self.t = t
        self.mu = mu
        self.Delta_s = Delta_s
        super().__init__(L_x, L_y, self._get_onsite(), self._get_hopping_x(),
                         self._get_hopping_y())
    def _get_onsite(self):
        return 1/2*((-self.mu)*np.kron(tau_z, sigma_0) + self.Delta_s*np.kron(tau_x, sigma_0) )
    def _get_hopping_x(self):
        return -1/2*self.t*np.kron(tau_z, sigma_0)
    def _get_hopping_y(self):
        return -1/2*self.t*np.kron(tau_z, sigma_0)
            
class TrivialSuperconductorPeriodicInY(PeriodicHamiltonianInY):
    def __init__(self, L_x:int, L_y:int, t:float, mu:float, Delta_s:float):
        self.t = t
        self.mu = mu
        self.Delta_s = Delta_s
        super().__init__(L_x, L_y, self._get_onsite(), self._get_hopping_x(),
                         self._get_hopping_y())
        self.t = t
        self.mu = mu
        self.Delta_s = Delta_s
    def _get_onsite(self):
        return 1/2*((-self.mu)*np.kron(tau_z, sigma_0) + self.Delta_s*np.kron(tau_x, sigma_0) )
    def _get_hopping_x(self):
        return -1/2*self.t*np.kron(tau_z, sigma_0)
    def _get_hopping_y(self):
        return -1/2*self.t*np.kron(tau_z, sigma_0)
            
class TrivialSuperconductorKY(Hamiltonian):
    r"""
    .. math::

        H_{A1us} = \frac{1}{2}\sum_k H_{A1us,k}
        
        H_{S,k} = \sum_n^L \vec{c}^\dagger_n\left[ 
            \xi_k\tau_z\sigma_0 + \Delta_0 \tau_x\sigma_0\right] +
            \sum_n^{L-1}\left(\vec{c}^\dagger_n(-t\tau_z\sigma_0 )\vec{c}_{n+1}
            + H.c. \right)
        
        \vec{c} = (c_{k,\uparrow}, c_{k,\downarrow},c^\dagger_{-k,\downarrow},-c^\dagger_{-k,\uparrow})^T
    
        \xi_k = -2tcos(k) - \mu
    """
    def __init__(self,  k:float, L_x:int, t:float, mu:float, Delta_s:float):
        self.t = t
        self.mu = mu
        self.Delta_s = Delta_s
        self.k = k
        super().__init__(L_x, 1, self._get_onsite(), self._get_hopping_x(),
                         np.zeros((4, 4)))
    def _get_onsite(self):
        chi_k = -2*self.t*np.cos(self.k)-self.mu
        return 1/2*( chi_k*np.kron(tau_z, sigma_0) +
                self.Delta_s*np.kron(tau_x, sigma_0) )
    def _get_hopping_x(self):
        return -1/2*self.t*np.kron(tau_z, sigma_0)       
        