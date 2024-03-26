# Imports
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
from scipy.optimize import curve_fit
from typing import Callable
from scipy.integrate import solve_ivp

# Constants
N_c = 3 # Number of colors
C_F = 3/2 # Fundamental Casimir in place of 4/3 in the large N_c limit
C_A = 3 # Adjoint Casimir
N_f = 5 # Number of flavors

Q_s_p = np.sqrt(0.2) # GeV
Q_s_pb = np.sqrt(0.6) # GeV
Lambda_QCD = 0.241 # GeV
x0 = 0.01 # Bjorken x

def r0(Q_s: float):
    """Computes the dipole size"""
    return 1/Q_s



def vnorms(v):
    """Computes the norm of a vector"""
    return np.linalg.norm(v)**2

def alpha_s(r: float,
            constant: bool = True):
    """Computes the running coupling constant at the scale r"""
    if constant:
        return 0.25
    beta_0 = (11*N_c - 2*N_f)/(6*np.pi)
    return 1/(beta_0*np.log(2/(r*Lambda_QCD)))

def K_run(r_vect: np.ndarray, # Dipole vector
          r1_vect: np.ndarray, # First dipole vector
          oneterm: bool = True #keeps only the last term of the kernel
          ):
    """Computes the kernel of the rcBK equation"""
    r2_vect = r_vect-r1_vect
    term3 = vnorms(r_vect)/(vnorms(r1_vect)*vnorms(r2_vect))
    
    if oneterm:
        return term3*alpha_s(vnorms(r_vect))*N_c/(2*(np.pi)**2)
    
    term1 = 1/vnorms(r1_vect)*(alpha_s(vnorms(r1_vect))/alpha_s(vnorms(r2_vect)) - 1)
    term2 = 1/vnorms(r2_vect)*(alpha_s(vnorms(r2_vect))/alpha_s(vnorms(r1_vect)) - 1)

    term = (term1 + term2 + term3)*alpha_s(vnorms(r_vect))*N_c/(2*(np.pi)**2)

    return term

def N_mv_0(r_vect:np.ndarray, # Dipole size
        x:float=x0, # Bjorken x 
        Q_s:float=Q_s_p # Saturation scale
        ):
    """McLerran-Venugopalan ansatz for the rcBK equation"""
    r = np.linalg.norm(r_vect)
    return 1-np.exp(-r**2*Q_s**2/4*np.log(1/(Lambda_QCD*r)+np.exp(1)))

def vectorize_function(R:np.ndarray, # Lattice of points
                        N_0:callable # Function N_0(r, x)
                ):
    """Take a function N_0(r, x) and returns a vector of the function values at the points in R"""
    N_0_vect = np.zeros(len(R))
    for i in range(len(R)):
        N_0_vect[i] = N_0(R[i])
    return N_0_vect


def integrand(N_sol: callable, #Solution function N(r, x)
            r_vect: np.ndarray, # Dipole vector
            r1_vect: np.ndarray, # First dipole vector
            x: float # Bjorken x
            ):
    """Integrand of the BK equation"""
    r = np.linalg.norm(r_vect)
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r_vect-r1_vect)
    K = K_run(r_vect, r1_vect)
    funcs = N_sol(r1, x) + N_sol(r2, x) - N_sol(r, x) - N_sol(r1, x)*N_sol(r2, x)
    return K*funcs


def find_closest_r_index(r_list: np.ndarray, 
                         r: float):
    """Finds the index of the closest r in the array r_list"""
    return np.argmin(np.abs(r_list-r))

# Calculating the integrals

def g(r: float,
      r1: float, 
      N_list: np.ndarray, 
      r_list: np.ndarray):
    """Computes the g function of the integrand present in the report"""
    if r == r1:
        return 0
    n_samples = 500
    thetas = np.linspace(0, 2*np.pi, n_samples)
    terms = np.array([])

    def N(rr):
        if rr < r_list[0]:
            factor = (rr/r_list[0])**2
            return factor*N_list[0]
        if rr > r_list[-1]:
            denominator = 1-np.exp((np.log(r_list[-1]))**2)
            numerator = 1-np.exp((np.log(rr))**2)
            ratio = numerator/denominator
            return ratio*N_list[-1]
            #return 1
        j = find_closest_r_index(r_list, rr)
        return N_list[j]
    

    for theta in thetas:
        denom = r**2 + r1**2 - 2*r*r1*np.cos(theta)
        numer = r**2*(N(r1)-N(r))
        term = 0
        if denom != 0 and numer != 0:
            term = numer/denom
        terms = np.append(terms, term)

    #Simpson's method
    delta = 2*np.pi/n_samples
    sum =0
    for i in range(n_samples):
        if i==0 or i==n_samples-1:
            sum += terms[i]
        else:
            if i%2 == 0:
                sum += 2*terms[i]
            else:
                sum += 4*terms[i]
    return delta*sum/3


def h(r: float,
      r1: float, 
      N_list: np.ndarray, 
      r_list: np.ndarray):
    """Computes the h function of the integrand present in the report"""
    if r == r1:
        return 0
    n_samples = 500
    thetas = np.linspace(0, 2*np.pi, n_samples)
    terms = np.array([])

    def N(rr):
        if rr < r_list[0]:
            factor = (rr/r_list[0])**2
            return factor*N_list[0]
        if rr > r_list[-1]:
            denominator = 1-np.exp((np.log(r_list[-1]))**2)
            numerator = 1-np.exp((np.log(rr))**2)
            ratio = numerator/denominator
            return ratio*N_list[-1]
            #return 1
        j = find_closest_r_index(r_list, rr)
        return N_list[j]
    
    for theta in thetas:
        denom = r**2 + r1**2 - 2*r*r1*np.cos(theta)
        r2 = np.sqrt(denom)
        numer = r**2*N(r2)*(1-N(r1))
        term = 0
        if denom != 0 and numer != 0:
            term = numer/denom
        terms = np.append(terms, term)

    delta = 2*np.pi/n_samples
    sum =0
    for i in range(n_samples):
        if i==0 or i==n_samples-1:
            sum += terms[i]
        else:
            if i%2 == 0:
                sum += 2*terms[i]
            else:
                sum += 4*terms[i]
    return delta*sum/3

## to be used only to test g(r, r1)

# def g_exact(r, r1):
#     if r != r1:
#         results, error = quad(lambda theta: r**2/(r**2 + r1**2 - 2*r*r1*np.cos(theta)), 0, 2*np.pi)
#     if r == r1:
#         return 0
#     return results


# def h_exact(r, r1, N_list, r_list):
#     if r != r1:
#         def N(rr):
#             if rr < r_list[0]:
#                 return 0
#             if rr > r_list[-1]:
#                 return 1
#             j = find_closest_r_index(r_list, rr)
#             return N_list[j]
#         results, error = quad(lambda theta: N(r*np.sqrt(1-2*np.cos(theta)*r1/r) )/(r**2 + r1**2 - 2*r*r1*np.cos(theta)), 0, 2*np.pi)
#     if r == r1:
#         return 0
#     return results




def integrals(N_list:np.ndarray, 
              r_list:np.ndarray
              ):
    """Computes the integrals of the IDE using the trapezoidal rule for each r
    
    
    Returns:
    array of length len(r_list) with the integrals which shuld correspond to dN(r_i)/dy
    """
    rho_list = [np.log(r) for r in r_list]
    bigterms = np.array([])
    for i in range(len(r_list)):
        r = r_list[i]
        terms = np.array([])
        for j in range(len(r_list)):
            r1 = r_list[j]
            term1 = g(r, r1, N_list, r_list)
            term2 = h(r, r1, N_list, r_list)
            term = term1 + term2
            terms = np.append(terms, term)
        delta = rho_list[1] - rho_list[0]
        sum = 0
        for k in range(len(r_list)):
            if k==0 or k==len(r_list)-1:
                sum += terms[k]
            else:
                if k%2 == 0:
                    sum += 2*terms[k]
                else:
                    sum += 4*terms[k]
        coeff = N_c*alpha_s(1)/(2*np.pi**2)
        bigterm = coeff*delta*sum/3
        print(f"for r={r} we have bigterm = {bigterm}")
        bigterms = np.append(bigterms, bigterm)
    return bigterms

#Solving the rcBK equation

def solve_rcBK_euler(N_0:callable, #Ansatz function
                     y0:float, #Initial rapidity
                     yf:float, #Final rapidity
                     dy:float, #Step size
                     r_list: np.ndarray #Lattice of points
                     ):
    """Solves the rcBK equation using the Euler method"""
    N_0_vect = vectorize_function(r_list, N_0)
    y = y0
    N_vect = N_0_vect
    y_vect = [y0]
    bigN = [N_vect]
    while y < yf:
        N_vect = N_vect + dy*integrals(N_vect, r_list)
        y = y + dy
        bigN.append(N_vect)
        y_vect.append(y)
        print(f"for y={y} it is done and bigN = \n {N_vect}")
    return y_vect, bigN

def solve_rcBK_runge_kutta(N_0:callable, #Ansatz function
                     y0:float, #Initial rapidity
                     yf:float, #Final rapidity
                     dy:float, #Step size
                     r_list: np.ndarray #Lattice of points
                     ):
    """Solves the rcBK equation using the Runge-Kutta method"""
    N_0_vect = vectorize_function(r_list, N_0)
    y = y0
    N_vect = N_0_vect
    y_vect = [y0]
    bigN = [N_vect]
    while y < yf:
        k1 = dy*integrals(N_vect, r_list)*alpha_s(1)/(2*np.pi)*N_c
        N_vect1 = N_vect + 0.5 * k1
        k2 = dy*integrals(N_vect1, r_list)*alpha_s(1)/(2*np.pi)*N_c
        N_vect2 = N_vect + 0.5 * k2
        k3 = dy*integrals(N_vect2, r_list)*alpha_s(1)/(2*np.pi)*N_c
        N_vect3 = N_vect + k3
        k4 = dy*integrals(N_vect3, r_list)*alpha_s(1)/(2*np.pi)*N_c
        N_vect = N_vect + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y = y + dy
        bigN.append(N_vect)
        y_vect.append(y)
        print(f"for y={y} it is done and bigN = \n {N_vect}")
    return y_vect, bigN

def solve_rcBK_ivp(N_0:callable, #Ansatz function
                     y0:float, #Initial rapidity
                     yf:float, #Final rapidity
                     r_list: np.ndarray #Lattice of points
                     ):
    """Solves the rcBK equation using the Runge-Kutta method using scipy solve ivp"""
    N_0_vect = vectorize_function(r_list, N_0)
    _g_ = lambda y, N_vect: integrals(N_vect, r_list)
    sol = solve_ivp(_g_, [y0, yf], N_0_vect, method='RK45', t_eval=np.linspace(y0, yf, 10), events=print_message, dense_output=False)
    return sol.t, sol.y



