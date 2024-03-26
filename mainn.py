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
    """McLerran-Venugopalan ansatz for the rcBK equation"""
    N_0_vect = np.zeros(len(R))
    for i in range(len(R)):
        N_0_vect[i] = N_0(R[i])
    return N_0_vect


def integrand(N_sol: callable, #Solution function N(r, x)
            r_vect: np.ndarray, # Dipole vector
            r1_vect: np.ndarray, # First dipole vector
            x: float # Bjorken x
            ):
    """Integrand of the IDE"""
    r = np.linalg.norm(r_vect)
    r1 = np.linalg.norm(r1_vect)
    r2 = np.linalg.norm(r_vect-r1_vect)
    K = K_run(r_vect, r1_vect)
    funcs = N_sol(r1, x) + N_sol(r2, x) - N_sol(r, x) - N_sol(r1, x)*N_sol(r2, x)
    return K*funcs


def lattice(Q_s: float, # Saturation scale
            M: int = 100 # Number of points in one axis 
            ):
    """Creates a lattice of points in the transverse plane of spacing 0.01/Q_s and size 100/Q_s"""
    # Lattice parameters
    
    r0 =  1/Q_s # Dipole size
    rr = r0*np.logspace(-2, 2, M)
    thetas = np.linspace(0, 2*np.pi, 30)
    R = np.array([[r*np.cos(theta), r*np.sin(theta)] for r in rr for theta in thetas])
    return R

def info_lattice(R):
    """Finds if the difference of two vectors in the lattice is in the lattice"""
    n_points = len(R)
    info_matrix = np.full((n_points, n_points), -1)  # Initialize with -1

    for i in range(n_points):   
        for j in range(n_points):
            diff_vect = R[i] - R[j]
            # Broadcasting to check if diff_vect matches any vector in R
            matches = np.all(R == diff_vect, axis=1)
            if np.any(matches):
                info_matrix[i, j] = np.where(matches)[0][0]

    return info_matrix

def K_matrix(R: np.ndarray, # Lattice of points
                oneterm: bool = True #keeps only the last term of the kernel
                ):
    """Computes the kernel matrix"""
    K = np.zeros((len(R), len(R)))
    for i in range(len(R)):
        for j in range(len(R)):
            K[i, j] = K_run(R[i], R[j], oneterm)
    return K

def _f_(y: float, # Bjorken y
        N_vect: np.ndarray, #Vector function N(x)=  [N(r1, x), N(r2, x), ...]
        R: np.ndarray # Lattice of points
            ):
    """Computes the f matrix"""
    x = x0*np.exp(-y) # Bjorken x
    n_points = len(R)
    K = K_matrix(R)
    info = info_lattice(R)
    output = np.zeros(n_points)
    for i in range(n_points):
        sum = 0
        for j in range(n_points):
            sum += K[i, j]*(N_vect[j]-N_vect[i])
            if info[i, j] != -1:
                sum += K[i,j]*N_vect[info[i, j]]*(1-N_vect[j])
            else:
                sum += K[i,j]*(1-N_vect[j])
        output[i] = sum/n_points
    return output



def print_message(y, N_vect):
    print(f"At y={y} it is done")
    return 1

def solve_rcBK(N_0, x0, xf, R):
    """Solves the rcBK equation using the Runge-Kutta method using scipy solve ivp"""
    N_0_vect = vectorize_function(R, N_0)
    yf = np.log(x0/xf)
    _g_ = lambda y, N_vect: _f_(y, N_vect, R)
    sol = solve_ivp(_g_, [0, yf], N_0_vect, method='RK45', t_eval=np.linspace(0, yf, 10), events=print_message, dense_output=False)
    return sol.t, sol.y

def solve_rcBK_euler(N_0, x0, xf, dy, R):
    """Solves the rcBK equation using the Euler method"""
    N_0_vect = vectorize_function(R, N_0)
    yf = np.log(x0/xf)
    y = 0
    N_vect = N_0_vect
    y_vect = [0]
    bigN = [N_vect]
    while y < yf:
        N_vect = N_vect + dy*_f_(y, N_vect, R)
        y = y + dy
        bigN.append(N_vect)
        y_vect.append(y)
        print(f"for y={y} it is done and bigN = \n {N_vect}")
    return y_vect, bigN   

#Define Lattice
# R = lattice(Q_s_p, M = 4)

# #Define Bjorken x
# x0 = 0.01
# yf = 0.5
# xf = 0.01/np.exp(yf)

# y_vect, bigN = solve_rcBK_euler(N_mv_0, x0, xf, 0.1, R)

# #Plot
# plt.plot(y_vect, [row[0] for row in bigN])
# plt.xlabel(r'$y$')
# plt.ylabel(r'$N(r, y)$')
# plt.title(r'$N(r, y)$ vs $y$')
# plt.show()

############################################################################################################

def find_closest_r_index(r_list, r):
    return np.argmin(np.abs(r_list-r))

def g(r, r1, N_list, r_list):
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


def h(r, r1, N_list, r_list):
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

#to be used only to test g(r, r1)
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




def integrals(N_list, r_list):
    """Computes the integrals of the IDE using the trapezoidal rule for each r
    
    
    Returns
    array of length len(r_list) with the integrals which shuld correspond to dN(r_i)/dy"""
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

def solve_rcBK_euler(N_0, y0, yf, dy, r_list):
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

def solve_rcBK_runge_kutta(N_0, y0, yf, dy, r_list):
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

def solve_rcBK_ivp(N_0, y0, yf, r_list):
    """Solves the rcBK equation using the Runge-Kutta method using scipy solve ivp"""
    N_0_vect = vectorize_function(r_list, N_0)
    _g_ = lambda y, N_vect: integrals(N_vect, r_list)
    sol = solve_ivp(_g_, [y0, yf], N_0_vect, method='RK45', t_eval=np.linspace(y0, yf, 10), events=print_message, dense_output=False)
    return sol.t, sol.y





    

    


        
            










            
    















############################################################################################################
def Monte_Carlo(N_sol: Callable, #Solution function N(r, x)
                r0: float, # Dipole size
                r_vect: np.ndarray, # Dipole vector
                x: float, # Bjorken x
                n_samples: int # Number of random samples
                ):
    """Monte Carlo integration of the dipole amplitude"""
    results = dict()
    for i in range(n_samples):
        rho = np.random.uniform(0, r0**2)
        theta = np.random.uniform(0, 2*np.pi)
        r1_vect = np.array([np.sqrt(rho)*np.cos(theta), np.sqrt(rho)*np.sin(theta)])
        r2_vect = r_vect-r1_vect
        I = integrand(N_sol, r_vect, r1_vect, x)
        results[r1_vect]=I*np.pi*r0**2
    return results
    return np.mean(results)

def Exact_integration(N_sol: Callable, #Solution function N(r, x)
                      r0: float, # Dipole size
                      r_vect: np.ndarray, # Dipole vector
                      x: float # Bjorken x
                      ):
    """Exact integration of the dipole amplitude"""
    results, error = dblquad(lambda theta1, r1: (r1*integrand(N_sol, r_vect, np.array([r1*np.cos(theta1), r1*np.sin(theta1)]), x)), 0, r0, lambda r1: 0, lambda r1: 2*np.pi)
    return results


# Create a function to fit the data using curve_fit
def fit_data(x_data, y_data):
    def custom_model(r, a, b):
        return N_0(a*(r-b))
    # Initial guess for the parameters (a, b, c)
    initial_guess = (1, 0)

    # Use curve_fit to fit the data to the custom model function
    params, covariance = curve_fit(custom_model, x_data, y_data, p0=initial_guess)

    # Extract the fitted parameters
    a_fit, b_fit = params

    # Return the fitted parameters and the fitted function
    def fitted_function(x):
        return custom_model(x, a_fit, b_fit)

    return fitted_function, params

def solve_rcBK_MC(N_0, dy, r0, x0, xf, r_vect, n_samples):
    """Solves the IDE using Monte Carlo integration"""
    x = x0
    N_sol = N_0
    N_sol_vect = [N_0(np.linalg.norm(r_vect), x0)]
    x_vect = [x0]
    yf = np.log(x0/xf)
    y = np.log(x0/x)
    while y < yf:
        inter_result = Monte_Carlo(N_sol, r0, r_vect, x, n_samples)
        N_sol_prime = N_sol_vect[0] + dy*inter_result
        N_sol_vect.insert(0, N_sol_prime)
        y = y + dy
        x = x0*np.exp(-y)
        x_vect.insert(0, x)
        #print(f"for r_vect={r_vect} and y={y} we have N(r,x+dy) = {N_sol_prime}")
    #print(f"---- N(r,xf) = {N_sol_vect[0]}-----")
    return x_vect, N_sol_vect

def solve_rcBK_MC_RK4(N_0, dy, r0, x0, xf, r_vect, n_samples):
    N_sol = N_0
    N_solutions = [N_0]
    x=x0
    x_vect = [x0]
    yf = np.log(x0/xf)
    y = np.log(x0/x)
    while y < yf:
        rhos = np.random.uniform(0, r0**2, n_samples)
        thetas = np.random.uniform(0, 2*np.pi, n_samples)
        r1_vects = [np.array([np.sqrt(rhos[i])*np.cos(thetas[i]), np.sqrt(rhos[i])*np.sin(thetas[i])]) for i in range(n_samples)]
        r1_vects_norms = np.sqrt(rhos)
        N_sol_values = [N_sol(r1,x) for r1 in r1_vects_norms]
        









def solve_rcBK_exact(N_0, dy, r0, x0, xf, r_vect):
    """Solves the IDE using exact integration"""
    x = x0
    N_sol = N_0
    N_sol_vect = [N_0(np.linalg.norm(r_vect), x0)]
    x_vect = [x0]
    yf = np.log(x0/xf)
    y = np.log(x0/x)
    while y < yf:
        inter_result = Exact_integration(N_sol, r0, r_vect, x)
        N_sol_prime = N_sol_vect[0] + dy*inter_result
        N_sol_vect.insert(0, N_sol_prime)
        y = y + dy
        x = x0*np.exp(-y)
        x_vect.insert(0, x)
        # print("x = ", x, "N = ", N_sol_prime, "inter_result = ", inter_result)
    return x_vect, N_sol_vect


def runge_kutta_step_MC(N_sol, r0, r_vect, x, dy, n_samples):
    """Perform one step of the Runge-Kutta method"""
    k1 = dy * Monte_Carlo(N_sol, r0, r_vect, x, n_samples)
    N_sol1 = lambda r, x: N_sol(r, x) + 0.5 * k1
    k2 = dy * Monte_Carlo(N_sol1, r0, r_vect, x + 0.5 * dy, n_samples)
    N_sol2 = lambda r, x: N_sol(r, x) + 0.5 * k2
    k3 = dy * Monte_Carlo(N_sol2, r0, r_vect, x + 0.5 * dy, n_samples)
    N_sol3 = lambda r, x: N_sol(r, x) + k3
    k4 = dy * Monte_Carlo(N_sol3, r0, r_vect, x + dy, n_samples)
    
    return N_sol(r_vect, x) + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

def solve_rcBK_MC_RK4(N_0, dy, r0, x0, xf, r_vect, n_samples):
    """Solves the IDE using the Runge-Kutta method (4th order)"""
    x = x0
    N_sol = N_0
    N_sol_vect = [N_0(np.linalg.norm(r_vect), x0)]
    x_vect = [x0]
    yf = np.log(x0/xf)
    y = np.log(x0/x)
    
    while x > xf:
        inter_result = runge_kutta_step_MC(N_sol, r0, r_vect, x, dy, n_samples)
        x = x * np.exp(-dy)
        N_sol_vect.insert(0, inter_result)
        x_vect.insert(0, x)
    
    return x_vect, N_sol_vect






# Gluon distribution function
    




def F_WW(x_2, k_t, n_samples, dx, r0, x0):
    coeff = C_F/(2*np.pi)*r0
    results = []
    def N(r_vect,x=x_2):
        return solve_rcBK_MC(N_0, dx, n_samples, r0, x0, x_2, r_vect)[1][-1]
    for i in range(n_samples):
        r_vect = np.random.uniform(0, r0, (1, 2))
        term1 = 1-(1-N(r_vect))**2
        term2 = np.exp(-1j*k_t*r_vect)
        results.append(term1*term2/(r_vect**2))
    
    return coeff*(np.sum(results).real)/n_samples


def F_fund(x2, k_t, n_samples, dy, r0, x0):
    """Computes the Fourier transform of the dipole amplitude"""
    coeff = N_c/(2*(np.pi)**2)*k_t**2
    def f(x,y):
        r_vect = np.array([x,y])
        x_vect, N_sol_vect = solve_rcBK_MC(N_0, dy, r0, x0, x2, r_vect, n_samples)
        return 1-N_sol_vect[0]
    x = np.linspace(-r0, r0, 100)
    y = np.linspace(-r0, r0, 100)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    return coeff*np.trapz(np.trapz(Z*np.exp(-1j*k_t*np.sqrt(X**2 + Y**2)), x), y)


    
    

#plot F_WW
def plot_F_fund(x_2, n_samples, dx, r0, x0):
    k_t_vect = np.logspace(-1, 2, 30)
    F_WW_vect = []
    for k_t in k_t_vect:
        F_WW_vect.append(F_fund(x_2, k_t, n_samples, dx, r0, x0))
    plt.plot(k_t_vect, F_WW_vect)
    plt.xlabel(r'$k_t$')
    plt.ylabel(r'$F_{WW}(x_2, k_t)$')
    plt.title(r'$F_{WW}(x_2, k_t)$ vs $k_t$')
    plt.savefig('F_WW.pdf')
    plt.show()






        

    
    
        









