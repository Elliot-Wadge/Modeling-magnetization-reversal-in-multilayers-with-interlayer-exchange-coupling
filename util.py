import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, brentq, minimize

def H_of_0(J1,J2,Ms1,Ms2):
    '''handle the special case of H = 0
    
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param Ms1: the saturation strength of one side of the interface in MA/m
    :type Ms1: float or np.ndarray
    :param Ms2: the saturation strength of the other side of the interface in MA/m
    :type Ms2: float or np.ndarray
    '''
    if J1 >= 2*J2:
        theta1 = np.pi
    else:
        theta1 = np.arctan(np.sqrt(1-(J1/(2*J2))**2)/((sum(Ms1))/(sum(Ms2)) - J1/(2*J2)))

    while theta1 < 0:
        #this is weird
        theta1 += np.pi
        
    
    return theta1


def check_callable(f,x):
    '''check if an object f is callable, if it is return f(x)'''
    if callable(f):
        return f(x)
    else:
        return f
    
def arrarize(a, N):
    '''if object is no iterable, update it to be an array of size N with value a'''
    if hasattr(a, "__iter__"):
        return a 
    return a*np.ones(N)