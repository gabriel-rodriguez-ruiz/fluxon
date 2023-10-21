# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 11:37:41 2023

@author: gabri
"""
import numpy as np

class Hamiltonian:
    """A class for 2D Bogoliubov-de-Gennes Hamiltonians.
    
        Parameters
        ----------
        L_x : int
            Number of sites in x-direction (horizontal).
        L_y : int
            Number of sites in y-direction (vertical).
        onsite : ndarray
            2x2 matrix representing the onsite term of the Hamiltonian.
        hopping_x : ndarray
            2x2 matrix representing the hopping term in x of the Hamiltonian.
        hopping_y : ndarray
            2x2 matrix representing the hopping term in y of the Hamiltonian.
    """
    # Pauli matrices (class variables)
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    tau_0 = np.eye(2)
    tau_x = np.array([[0, 1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    tau_z = np.array([[1, 0], [0, -1]])
    def __init__(self, L_x:int, L_y:int, onsite,
                 hopping_x, hopping_y):
        self.L_x = L_x
        self.L_y = L_y
        self.onsite = onsite
        self.hopping_x = hopping_x
        self.hopping_y = hopping_y
        self.matrix = self._get_matrix()
    def _index(self, i:int , j:int, alpha:int):    
        #protected method, accesible from derived class but not from object
        r"""Return the index of basis vector given the site (i,j)
        and spin index alpha in {0,1,2,3} for i in {1, ..., L_x} and
        j in {1, ..., L_y}. The site (1,1) corresponds to the lower left real
        space position.
        
        .. math ::
            \text{Basis vector} = 
           (c_{11}, c_{12}, ..., c_{1L_y}, c_{21}, ..., c_{L_xL_y})^T
           
           \text{index}(i,j,\alpha,L_x,L_y) = \alpha + 4\left(L_y(i-1) +
                                              + j-1\right)
           
           \text{real space}
           
           (c_{1L_y} &... c_{L_xL_y})
                            
           (c_{11} &... c_{L_x1})
         Parameters
         ----------
         i : int
             Site index in x direction. 1<=i<=L_x
         j : int
             Positive site index in y direction. 1<=j<=L_y
         alpha : int
             Spin index. 0<=alpha<=3
        """
        if (i>self.L_x or j>self.L_y):
            raise Exception("Site index should not be greater than \
                            samplesize.")
        return alpha + 4*( self.L_y*(i-1) + j-1 )
    def _get_matrix(self):
        """
        Matrix of the BdG-Hamiltonian.
        """
        L_x = self.L_x
        L_y = self.L_y
        M = np.zeros((4*L_x*L_y, 4*L_x*L_y), dtype=complex)
        #onsite
        for i in range(1, L_x+1):    
            for j in range(1, L_y+1):
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i , j, alpha), self._index(i, j, beta)]\
                            = 1/2*self.onsite[alpha, beta]
                            # factor 1/2 in the diagonal because I multiplicate
                            # with the transpose conjugate matrix
        #hopping_x
        for i in range(1, L_x):
            for j in range(1, L_y+1):    
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i+1, j, beta)]\
                        = self.hopping_x[alpha, beta]
        #hopping_y
        for i in range(1, L_x+1):
            for j in range(1, L_y): 
                for alpha in range(4):
                    for beta in range(4):
                        M[self._index(i, j, alpha), self._index(i, j+1, beta)]\
                        = self.hopping_y[alpha, beta]
        return M + M.conj().T
 