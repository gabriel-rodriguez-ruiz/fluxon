import numpy as np
import matplotlib.pyplot as plt
import scipy
from phase_functions import phase_soliton_antisoliton_S_around_zero, phase_soliton_antisoliton
from superconductor import TrivialSparseSuperconductor, \
                            A1usSparseSuperconductor                            
from junction import PeriodicJunction

L_x = 300
L_y = 300
t = 1
t_J = t/5
Delta_s_Trivial = t/5
Delta_p_A1us = t/5
Delta_s_A1us = t/20
mu = -2*t
n = 12      #number of eigenvalues in sparse diagonalization
phi_external = 0.
phi_eq = 0.12*2*np.pi   #0.053*2*np.pi    #0.14*2*np.pi
y = np.arange(1, L_y+1)
L_values = np.linspace(1, 10, 10, dtype=int)

eigenvalues = []

for L_value in L_values:
    y_0 = (L_y-L_value)//2
    y_1 = (L_y+L_value)//2
    # Phi = phase_soliton_antisoliton_S_around_zero(phi_external, phi_eq, y, y_0, y_1)
    Phi = phase_soliton_antisoliton(phi_external, y, y_0, y_1)
    S_A1us = A1usSparseSuperconductor(L_x, L_y, t, mu, Delta_s_A1us, Delta_p_A1us)
    # S_Trivial = TrivialSparseSuperconductor(L_x, L_y, t, mu, Delta_s_Trivial)
    # J = PeriodicJunction(S_A1us, S_Trivial, t_J, Phi)
    J = PeriodicJunction(S_A1us, S_A1us, t_J, Phi)
    eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(J.matrix, k=n, sigma=0) 
    eigenvalues_sparse.sort()
    eigenvalues.append(eigenvalues_sparse)
    print(L_value)
index = np.arange(n)
E_numerical = []
for j in index:
    E_numerical.append(np.array([eigenvalues[i][j] for i in range(len(L_values))]))

#%% Plotting

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


fig, ax = plt.subplots()
# I remove the L=100 distance and plot only zero-energy states
ax.plot(L_values, E_numerical[n//2], "o", label="Numerical zero state")
ax.plot(L_values, E_numerical[n//2+1], "*")
ax.plot(L_values, E_numerical[n//2+2], ".")
ax.plot(L_values, E_numerical[n//2+3], ".")
ax.plot(L_values, E_numerical[n//2+4], ".")
ax.plot(L_values, E_numerical[n//2+5], ".")

ax.set_xlabel(r"$L$")
plt.yscale('log')
ax.set_ylabel("E")

from analytical_solution import Kappa

# m_0 = t_J**2/Delta_s_Trivial
m_0 = t_J/2


def positive_energy(L, m_0):
    kappa_value = Kappa(m_0=m_0, Delta=Delta_p_A1us, L=L_value)
    # kappa_value = m_0
    return m_0*np.exp(-kappa_value*L)

E = []
for L_value in L_values:
    Energy = positive_energy(L=L_value, m_0=m_0)
    E.append(Energy[0])
    # E.append(Energy)

E_analytical = np.array([E[i] for i in range(len(L_values))])
# ax.plot(L_values, E_analytical, "ok", label="Analytical")
# ax.plot(0, m_0, "ok", label="Analytical")

# E_numerics = np.array([#E_numerical[n//2+3][0],
#             E_numerical[n//2+1][1],
#             E_numerical[n//2+1][2],
#             E_numerical[n//2+1][3],
#             E_numerical[n//2+1][4],
#             E_numerical[n//2+1][5],
#             E_numerical[n//2+1][6],
#             E_numerical[n//2+1][7],
#             E_numerical[n//2+1][8],
#             E_numerical[n//2+1][9]])

E_numerics = np.array([E_numerical[n//2+3][0],
            E_numerical[n//2+3][1],
            E_numerical[n//2+3][2],
            E_numerical[n//2+3][3],
            E_numerical[n//2+4][4]])

# ax.plot(L_values, E_analytical, "o")
m_numerical, b_numerical = np.polyfit(L_values, np.log(E_numerics), 1)
# m_analytical, b_analytical = np.polyfit(L_values, np.log(E_analytical), 1)
# m_analytical, b_analytical = np.polyfit(L_values, np.log(E_analytical), 1)
# ax.plot(L_values, E_numerics)
x = np.linspace(2, 10)
# ax.plot(x, np.exp(b_numerical)*np.exp(m_numerical*x), label=f"{m_numerical:.2}L{b_numerical:.2}")
# plt.title(r"$\phi_{eq}=$"+f"{phi_eq:.2}, Delta={Delta_p_A1us}")
plt.tight_layout()
plt.savefig('demo.pdf', transparent=True)