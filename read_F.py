
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mainn import *


# Specify the path to your text file
file_path_proton = 'ww_mv_qs02_02_af1_N1.txt'
file_path_lead   = 'ww_mv_qs02_02_af1_N3.txt'

# Use numpy.loadtxt to read data from the text file
data_proton = np.genfromtxt(file_path_proton)
data_lead   = np.genfromtxt(file_path_lead)

y2_values = np.unique(data_proton[:, 0])
kt_values = np.unique(data_proton[:, 1])

n_kt = len(kt_values)
max_kt = max(kt_values)


def find_closest_kt_index(kt):
    return np.argmin(np.abs(kt_values - kt))

############################################################################################################

#Functions from the data file

def F(y2, data):
    if y2 not in y2_values:
        raise ValueError(f'y2 = {y2} not in the y2 values data')
    values = data[data[:, 0] == y2]
    def F_prime(kt):
        i = find_closest_kt_index(kt)
        return values[i, 2]/(4*np.pi**2)
    
    return F_prime

def F_fund(y2, data):
    """Returns the fundamental F function for a given y2 value. """
    if y2 not in y2_values:
        raise ValueError(f'y2 = {y2} not in the y2 values data')
    values = data[data[:, 0] == y2]
    def F_prime(kt):
        i = find_closest_kt_index(kt)
        return (N_c/(8*np.pi**4))*kt**2*values[i, 2]
    
    return F_prime

def F_adj(y2, data):
    """Returns the adjoint F function for a given y2 value. """
    if y2 not in y2_values:
        raise ValueError(f'y2 = {y2} not in the y2 values data')
    values = data[data[:, 0] == y2]
    def F_prime(kt):
        i = find_closest_kt_index(kt)
        return (C_F/(8*np.pi**4))*kt**2*values[i, 3]
    
    return F_prime

def F_WW(y2, data):
    """Returns the WW F function for a given y2 value. """
    if y2 not in y2_values:
        raise ValueError(f'y2 = {y2} not in the y2 values data')
    values = data[data[:, 0] == y2]
    def F_prime(kt):
        i = find_closest_kt_index(kt)
        return (C_F/(2*np.pi**4))*values[i, 4]
    
    return F_prime


############################################################################################################

# def F_qg_2(y2, n_sample=5000):
#     F1 = F(y2)
#     F_WW1 = F_WW(y2)
#     results = []
#     for kt in kt_values:
#         kt_vect = np.array([kt, 0])
#         sum = 0
#         for i in range(n_sample):
#             rho = np.random.rand() * (max_kt)**2
#             r = np.sqrt(rho)
#             theta = np.random.rand() * 2 * np.pi
#             qt_vect = np.array([r * np.cos(theta), r * np.sin(theta)])
#             sum += F1(np.linalg.norm(kt_vect - qt_vect)) * F_WW1(r)
#         results.append(np.pi*max_kt**2*sum/n_sample)

#     def F_qg_2_prime(kt):
#         i = find_closest_kt_index(kt)
#         return results[i]
#     return F_qg_2_prime
   

# def F_gg_1(y2, n_sample=5000):
#     F1 = F(y2)
#     F_fund1 = F_fund(y2)
#     results = []
#     for kt in kt_values:
#         kt_vect = np.array([kt, 0])
#         sum = 0
#         for i in range(n_sample):
#             rho = np.random.rand() * (max_kt)**2
#             r = np.sqrt(rho)
#             theta = np.random.rand() * 2 * np.pi
#             qt_vect = np.array([r * np.cos(theta), r * np.sin(theta)])
#             sum += F1(np.linalg.norm(kt_vect - qt_vect)) * F_fund1(r)
#         results.append(np.pi*max_kt**2*sum/n_sample)
    
#     def F_gg_1_prime(kt):
#         i = find_closest_kt_index(kt)
#         return results[i]
    
#     return F_gg_1_prime


# def F_gg_6(y2, n_sample=5000):
#     F1 = F(y2)
#     F_WW1 = F_WW(y2)
#     results = []
#     for kt in kt_values:
#         kt_vect = np.array([kt, 0])
#         sum = 0
#         for i in range(n_sample):
#             rho, rho1 = np.random.rand() * (max_kt)**2, np.random.rand() * (max_kt)**2
#             r, r1 = np.sqrt(rho), np.sqrt(rho1)
#             theta, theta1 = np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi
#             qt_vect = np.array([r * np.cos(theta), r * np.sin(theta)])
#             qt_vect1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])
#             sum += F_WW1(r)*F1(r1)*F1(np.linalg.norm(kt_vect - qt_vect-qt_vect1))
#         results.append((np.pi*max_kt**2)**2*sum/n_sample)
    
#     def F_gg_6_prime(kt):
#         i = find_closest_kt_index(kt)
#         return results[i]
    
#     return F_gg_6_prime


############################################################################################################

# from scipy.signal import convolve2d

# def F_qg_2_conv(y2):
#     x_values = kt_values
#     y_values = kt_values
#     x_mesh, y_mesh = np.meshgrid(x_values, y_values)

#     F1 = F(y2)
#     def F11(x, y):
#         return F1(np.sqrt(x**2 + y**2))
    
#     F_WW1 = F_WW(y2)
#     def F_WW11(x, y):
#         return F_WW1(np.sqrt(x**2 + y**2))

#     f_values, g_values = np.zeros((n_kt, n_kt)), np.zeros((n_kt, n_kt))
#     for i in range(n_kt):
#         for j in range(n_kt):
#             f_values[i, j] = F11(x_values[i], y_values[j])
#             g_values[i, j] = F_WW11(x_values[i], y_values[j])

#     # Perform the 2D convolution
#     convolution_result = convolve2d(f_values, g_values, mode='same', boundary='symm')

#     # Define a function for the convolution result
#     def convolution_function(x, y):
#         x_index = np.argmin(np.abs(x_values - x))
#         y_index = np.argmin(np.abs(y_values - y))
#         return convolution_result[y_index, x_index]
    
#     return convolution_function

# def F_gg_1_conv(y2):
#     x_values = kt_values
#     y_values = kt_values
#     x_mesh, y_mesh = np.meshgrid(x_values, y_values)

#     F1 = F(y2)
#     def F11(x, y):
#         return F1(np.sqrt(x**2 + y**2))
    
#     F_fund1 = F_fund(y2)
#     def F_fund11(x, y):
#         return F_fund1(np.sqrt(x**2 + y**2))

#     f_values, g_values = np.zeros((n_kt, n_kt)), np.zeros((n_kt, n_kt))
#     for i in range(n_kt):
#         for j in range(n_kt):
#             f_values[i, j] = F11(x_values[i], y_values[j])
#             g_values[i, j] = F_fund11(x_values[i], y_values[j])

#     # Perform the 2D convolution
#     convolution_result = convolve2d(f_values, g_values, mode='same', boundary='symm')

#     # Define a function for the convolution result
#     def convolution_function(x, y):
#         x_index = np.argmin(np.abs(x_values - x))
#         y_index = np.argmin(np.abs(y_values - y))
#         return convolution_result[y_index, x_index]
    
#     return convolution_function



############################################################################################################


from scipy.integrate import dblquad

def F_qg_1(y2, data):
    return F_fund(y2, data)

def F_qg_2(y2, data):
    """Computes the TMD quark-gluon correlation function F_qg^(2)."""
    print(f"----------------Computing F_qg^(2) for y2 = {y2}-----------------" )
    F1 = F(y2, data)
    F_WW1 = F_WW(y2, data)
    results = []
    
    for kt in kt_values:
        def integrand(theta, r):
            r_prime = np.sqrt(kt**2 + r**2 - 2*kt*r*np.cos(theta))
            return r*F1(r_prime)*F_WW1(r)
        
        res, error = dblquad(integrand, 0, max_kt, lambda x: 0, lambda x: 2*np.pi)
        results.append(res)
        print(f"kt = {kt} done, result = {results[-1]}")

    def F_qg_2_prime(kt):
        i = find_closest_kt_index(kt)
        return results[i]
    print(f"----------------Done computing F_qg^(2) for y2 = {y2}-----------------" )
    return F_qg_2_prime

def F_gg_1(y2, data):
    """Computes the TMD gluon-gluon correlation function F_gg^(1)."""
    print(f"----------------Computing F_gg^(1) for y2 = {y2}-----------------" )
    F1 = F(y2, data)
    F_fund1 = F_fund(y2, data)
    results = []
    
    for kt in kt_values:
        def integrand(theta, r):
            r_prime = np.sqrt(kt**2 + r**2 - 2*kt*r*np.cos(theta))
            return r*F1(r_prime)*F_fund1(r)
        
        res, error = dblquad(integrand, 0, max_kt, lambda x: 0, lambda x: 2*np.pi)
        results.append(res)
        print(f"kt = {kt} done, result = {results[-1]}")

    def F_gg_1_prime(kt):
        i = find_closest_kt_index(kt)
        return results[i]
    print(f"----------------Done computing F_gg^(1) for y2 = {y2}-----------------" )
    return F_gg_1_prime

def F_gg_2(y2, data):
    FF = F_gg_1(y2, data)
    def F_prime(kt):
        return FF(kt)-F_adj(y2, data)(kt)
    return F_prime


#intermediate function for F_gg_6
def F_F(y2, data):
    """Computes the convolution of F and F"""
    print(f"----------------Computing F_F for y2 = {y2}-----------------" )
    F1 = F(y2, data)
    F_fund1 = F1
    results = []
    
    for kt in kt_values:
        def integrand(theta, r):
            r_prime = np.sqrt(kt**2 + r**2 - 2*kt*r*np.cos(theta))
            return r*F1(r_prime)*F_fund1(r)
        
        res, error = dblquad(integrand, 0, max_kt, lambda x: 0, lambda x: 2*np.pi)
        results.append(res)
        print(f"kt = {kt} done, result = {results[-1]}")

    def F_F_prime(kt):
        i = find_closest_kt_index(kt)
        return results[i]
    print(f"----------------Done computing F_F for y2 = {y2}-----------------" )
    return F_F_prime



def F_gg_6(y2, data):
    """Computes the TMD gluon-gluon correlation function F_gg^(6)."""
    print(f"----------------Computing F_gg^(6) for y2 = {y2}-----------------")
    F1 = F_F(y2, data)
    Fww = F_WW(y2, data)
    results = []
    
    for kt in kt_values:
        def integrand(theta, r):
            #YOU NEED TO INTROUCE THE DEPENDANCE ON Q_T' HERE
            r_prime = np.sqrt(kt**2 + r**2 - 2*kt*r*np.cos(theta))
            return r*Fww(r)*F1(r_prime)
        
        res, error = dblquad(integrand, 0, max_kt, lambda x: 0, lambda x: 2*np.pi)

        results.append(res)
        print(f"kt = {kt} done, result = {results[-1]}")

    def F_gg_6_prime(kt):
        i = find_closest_kt_index(kt)
        return results[i]
    print(f"----------------Done computing F_gg^(6) for y2 = {y2}-----------------" )
    return F_gg_6_prime

def TMDs(y2, data):
    return F_qg_1(y2, data), F_qg_2(y2, data), F_gg_1(y2, data), F_gg_2(y2, data), F_gg_6(y2, data)












    

    
