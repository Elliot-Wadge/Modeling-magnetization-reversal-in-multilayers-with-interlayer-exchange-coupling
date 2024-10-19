import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, brentq, minimize
import scipy


def energy_distribution(H, J1, J2, N1, N2, A1, A2, Ms1, Ms2,d):
    '''function to return the optimal thetas by minimizng the energy at a given field
    :param float H: the field strength in tesla
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param int N1: number of layers on one side of the interface
    :param int N2: number of layers on the other side of the interface
    :param A1: the exchange stiffness on one side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param A1: the exchange stiffness on the other side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param Ms1: the saturation strength of one side of the interface in MA/m
    :type Ms1: float or np.ndarray
    :param Ms2: the saturation strength of the other side of the interface in MA/m
    :type Ms2: float or np.ndarray
    :param d: the atomic spacing in units of 10^-8'''
    
    # unit conversions to match the continuous model
    Msd1 = Ms1*d*0.1
    Msd2 = Ms2*d*0.1
    Ad1 = A1*100/(d)
    Ad2 = A2*100/(d)
    ini_thetas = np.concatenate((np.arange(1,N1+1,1), np.arange(1,N2+1,1)))

    thetas_opt = minimize(energy_asymmetric, ini_thetas, args = (J1, J2, H, N1, Ad1, Ad2, Msd1, Msd2), tol = 0.0001).x

    return thetas_opt


def energy_asymmetric(thetas,J1, J2, H, N1, Ad1, Ad2, Msd1, Msd2):
    '''calculate the energy of a given system provided the magnetic angles, and system properties
    :param np.ndarray thetas: an array of angles corresponding to magnetic angle as defined in our paper
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float H: the field strength in tesla
    :param int N1: number of layers on one side of the interface
    :param int N2: number of layers on the other side of the interface
    :param Ad1: the exchange stiffness on one side of the interface times atomic spacing
    :type Ad1: float or np.ndarray
    :param Ad1: he exchange stiffness on one side of the interface times atomic spacing
    :type Ad1: float or np.ndarray
    :param Msd1: the saturation strength of one side of the interface times atomic spacing
    :type Msd1: float or np.ndarray
    :param Ms2: the saturation strength of the other side of the interface times atomic spacing
    :type Ms2: float or np.ndarray'''
    
    E_RKKY = J1 * np.cos(thetas[N1-1] - thetas[N1]) + J2*np.cos(thetas[N1-1] - thetas[N1])**2
    E_ex = -2*(np.sum(Ad1*np.cos(thetas[:N1-1] - thetas[1:N1])) + np.sum(Ad2*np.cos(thetas[N1:-1] - thetas[N1+1:])))
    E_ZCo = -H*np.sum(Msd1*np.cos(thetas[:N1])) - H*np.sum(Msd2*np.cos(thetas[N1:]))
    # print(f"E_RKKY = {E_RKKY}\nE_ex = {E_ex}\nE_Z = {E_ZCo}\ntotal={E_RKKY + E_ex + E_ZCo}")
    return E_RKKY + E_ex + E_ZCo



def energy_M(field, J1, J2, N1, N2, A1, A2, Ms1, Ms2,d, tol=1e-4):
    '''
    take in a range of field strengths and other parameters and return the M(H) curve
    
    :param np.ndarray field: the field strengths in tesla which you want to calculate M(H) in T
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param int N1: number of layers on one side of the interface
    :param int N2: number of layers on the other side of the interface
    :param A1: the exchange stiffness on one side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param A1: the exchange stiffness on the other side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param Ms1: the saturation strength of one side of the interface in MA/m
    :type Ms1: float or np.ndarray
    :param Ms2: the saturation strength of the other side of the interface in MA/m
    :type Ms2: float or np.ndarray
    :param d: the atomic spacing in units of 10^-8 '''
    
    
    ret = np.ones(len(field))
    # unit conversions to match the continuous model
    Msd1 = Ms1*d*0.1
    Msd2 = Ms2*d*0.1
    Ad1 = A1*100/(d)
    Ad2 = A2*100/(d)
    ini_thetas = np.concatenate((np.arange(1,N1+1,1), np.arange(1,N2+1,1)))
    # ini_thetas = np.ones(N1+N2)
    for i,H in enumerate(field):
    
        #print(J1, J2, field, N1, N2, Ad1, Ad2, Msd1, Msd2)
        #minimize the angles
        # bounds=zip(np.zeros(N1+N2), np.pi*np.ones(N1+N2))
        thetas_opt = minimize(energy_asymmetric, ini_thetas, args = (J1, J2, H, N1, Ad1, Ad2, Msd1, Msd2), tol = tol).x
        ini_thetas = thetas_opt
        # print(thetas_opt)

        if hasattr(Ms1, "__iter__"):
            Ms1_sum = np.sum(Ms1)
        else:
            Ms1_sum = N1*Ms1

        if hasattr(Ms2, "__iter__"):
            Ms2_sum = np.sum(Ms2)
        else:
            Ms2_sum = N2*Ms2


        mag = 1/(Ms1_sum + Ms2_sum) * (np.sum(Ms1*np.cos(thetas_opt[:N1])) + np.sum(Ms2*np.cos(thetas_opt[N1:])))
        if abs(1-mag) < 0.001:
            break

        ret[i] = mag
        
    return ret


