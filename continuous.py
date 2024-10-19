import numpy as np
from scipy.integrate import solve_ivp
from continuous_root import extrema, boundary_max, root_find, root_v2, get_A_transitions
import scipy
from util import check_callable, H_of_0

def derivatives(x,y,H,A,Ms):
    '''derivatives where 0 is taken to be the interface and d is taken to be the edge of the 
    layers, defined to be compatable with scipy.solve_ivp
    
    :param x: to be provided as fun(x,y) to scipy.solve_ivp
    :param y: the derivative state to be provided as fun(x,y) to scipy.solve_ivp
    :param float H: the field strength in tesla
    :param A: the exchange stiffness in the bulk to be propegated 10^-11 J/m
    :type A: float or callable
    :param Ms: the saturation strength in the bulk to be propegated in MA/m
    :type Ms: float or callable'''
    dw = (H*check_callable(Ms,x))/(2*check_callable(A,x)*10)*np.sin(y[0])
    dphi = y[1]
    
    return np.array([dphi, dw])

def negative_derivatives(x, y, H, A, Ms):
    '''return the derivatives of the system of equations describing the phi(x),
    where 0 is taken to be the edge of the layers and d is the interface, defined to be compatable with scipy.solve_ivp
    
    :param x: to be provided as fun(x,y) to scipy.solve_ivp
    :param y: the derivative state to be provided as fun(x,y) to scipy.solve_ivp
    :param float H: the field strength in tesla
    :param A: the exchange stiffness in the bulk to be propegated 10^-11 J/m
    :type A: float or callable
    :param Ms: the saturation strength in the bulk to be propegated in MA/m
    :type Ms: float or callable
    '''
    dw = (H*check_callable(Ms,x))/(2*check_callable(A,x)*10)*np.sin(y[0])
    dphi = y[1]
    if dw == 0 and dphi == 0:
        # known unstable solution give nudge
        dw += 1e-7
    return -np.array([dphi, dw])

def A_negative_boundary_conditions(phi2, phi1, w, J1, J2, A):
    '''the boundary conditiong at the outermost layer in the asymmetric case, returns 0 when boundary condition is satisfied
    
    :param phi2 float: angle in radians of magnetic moment second from the edge
    :param phi1 float: angle in radians of magenetic moment closest to the edge
    :param w foat: first derivative at the edge
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A1 float: the exchange stiffness in the bulk at the edge in 10^-11 J/m
    '''
    return -2*A*10*w - J1*np.sin(phi1 + phi2) - J2*np.sin(2*(phi1+phi2))




def propegate_forward(deriv, phi0, w0, H, A, Ms, transitions):
    '''function to propegate to the differential forward through the bulk while handling changes in 
    the value of A
    
    :param callable deriv: the derivate function in the bulk, in this case either derivative or negative_derivatives
    :param float phi0: initial guess for starting angle in radians at one side of the device 
    :param float w0: initial guess for the starting derivative in radian/unit distance (not actually a guess set to 0)
    :param float H: field strength in Tesla
    :param Ms: the saturation strength in the bulk to be propegated in MA/m
    :type Ms: float or callable
    :param iterable transition: a list containing the points at which A(x) changes, this seems not ideal
    
    '''
    x_prev = 0
    w_prev = w0
    phi_prev = phi0
    phi = []
    w = []
    x_ = []
    for transition in transitions:
        # range to solve the equation over
        tspan = (x_prev, transition)
        # calculate the value of the derivative across the interface
        new_w = w_prev * check_callable(A, x_prev)/check_callable(A, transition)
        # return the value of A in this region
        A_ = check_callable(A, transition)
        # solve the IVP with the calculated initial values and A
        sol = solve_ivp(deriv, tspan, [phi_prev, new_w], args=[H, A_, Ms],
                        t_eval=np.linspace(x_prev, transition, (10)), rtol=1e-8)
        # update the values for the next loop
        x_prev = transition
        w_prev = sol.y[1][-1]
        phi_prev = sol.y[0][-1]
        x_.append(sol.t)
        phi.append(sol.y[0])
        w.append(sol.y[1])
    
    # combine all the regions together
    phi = np.concatenate(phi)    
    w = np.concatenate(w)
    x_ = np.concatenate(x_)
    # return distribution
    return x_, phi, w


@np.vectorize
def continuous_distribution(phi0, H, J1, J2, A1, A2, Ms1, Ms2, half_thickness1, half_thickness2, show=False, ret_split=False):
    '''propegate an initial guess to the other end of the device and return a list of x values, angles, and derivative of angles
    
    :param float phi0: initial guess for starting angle
    :param float H: external field strength in Tesla
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
    :param d: the atomic spacing in units of 10^-8
    '''
    
    
    transitions1 = get_A_transitions(A1, half_thickness1)
    x1, phi1, w1 = propegate_forward(negative_derivatives, phi0, 0, H, A1, Ms1, transitions1)
    
    A10 = check_callable(A1,half_thickness1)
    A20 = check_callable(A2,0)
    
    max_phi2 = boundary_max(J1,J2,phi1[-1])
    w20 = (A10/A20 * w1[-1])
    x2 = np.linspace(0, half_thickness2, 100)


    # there's an assumption that w1[-1] < 0 for this to work, could be made more general but would require more code
    # if A_negative_boundary_conditions(max_phi2, phi1[-1], w1[-1], J1, J2, A10) > 0:
        
    #     x = np.concatenate((x1,x2+x1[-1]))
    #     phi = np.zeros(len(x))
    #     w = np.zeros(len(x))
        
    #     return x, phi, w
    
    
    lower = 0
    upper = np.pi - phi1[-1]
    
    
    phi2_roots = root_find(A_negative_boundary_conditions, [lower,upper], phi1[-1], w1[-1], J1, J2, A10)
    
    if show:
        print(f"boundary roots = {phi2_roots}")

            
    index = None
    min_w2 = np.inf
    
    transitions2 = get_A_transitions(A2, half_thickness2)
    
    for i,root in enumerate(phi2_roots):
        x_, phi2_temp, w2_temp = propegate_forward(derivatives, root, w20, H, A2, Ms2, transitions2)
        
        # elif abs(sol2_temp.y[1][-1]) < tol:
        #     index = i
        #     break

        if abs(w2_temp[-1]) < min_w2  and np.all(phi2_temp <= np.pi) and np.all(phi2_temp >= 0):
            index = i
            min_w2 = abs(w2_temp[-1])

    
        
    
    if index is not None:
        x2, phi2, w2 = propegate_forward(derivatives, phi2_roots[index], w20, H, A2, Ms2, transitions2)
    else:
        raise(ValueError("initial angle could not be propagated"))
    
    
    
    x = np.concatenate((x1,x2+x1[-1]))
    phi = np.concatenate((phi1,phi2))
    w = np.concatenate((w1,w2))

    if ret_split:
        return x, phi, w, len(x1)
    
    return x, phi, w





@np.vectorize
def continuous_shoot(phi0, H, J1, J2, A1, A2, Ms1, Ms2, half_thickness1, half_thickness2):
    '''propegate an initial guess to the other end of the device and return the boundary condition,
    returns zero when the boundary condition is satisfied
    
    :param float phi0: initial guess for starting angle
    :param float H: external field strength in Tesla
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
    :param d: the atomic spacing in units of 10^-8
    '''
    try:
        x, phi, w = continuous_distribution(phi0, H, J1, J2, A1, A2, Ms1, Ms2, half_thickness1, half_thickness2)
        return abs(w[-1])
    except ValueError:
        return 0




def continuous_M(fields, J1, J2, A1, A2, Ms1, Ms2, ht1,ht2, zero_bound=1e-4, sorted=True, tol=1e-3, step=1e-2, min_tol=1e-5):
    '''take in a range of field strengths and other parameters and return the M(H) curve using continuous asymmetric model
    
    :param np.ndarray field: the field strengths in tesla which you want to calculate M(H) in T
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A1: the exchange stiffness on one side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param A1: the exchange stiffness on the other side of the interface in 10^-11 J/m
    :type A1: float or np.ndarray
    :param Ms1: the saturation strength of one side of the interface in MA/m
    :type Ms1: float or np.ndarray
    :param Ms2: the saturation strength of the other side of the interface in MA/m
    :type Ms2: float or np.ndarray
    :param float d: the atomic spacing in units of 10^-8
    :param float ht1: thickness of side 1 of device
    :param float ht2: thickness of side 2 of device
    :param zero_bound: not in use right now
    :param bool sorted: if True then fields need to be provided in asceneding order, allows for speed up in computation by bounding by the previous solution
    :param float tol: provided to root_v2
    :param float min_tol: min_tol provided to root_v2
    :param float step: step size used by root finder, smaller steps more accurate but slows down solving'''
    # initialize array for storing return array
    
    mag = np.zeros(len(fields))
   
    # if J1 < 2*J2:
    #     upper_bound = 1/2*np.arccos(-J1/(2*J2))
    # else:
    upper_bound = np.pi
        
        
    for index, H in enumerate(fields):
        
        if H == 0:
            #special case where H = 0
            phi0 = H_of_0(J1,J2,Ms1,Ms2,ht1,ht2)
            
        else:
            # upon nearing saturation the solution becomes zero the solver requires
            # a point on either side of axis so this causes
            # error the except handles and sets phi0 to zero because we are at saturation
            
                
            phi0 = root_v2(continuous_shoot, 1e-3, upper_bound+2*tol, tol=tol, step=step, min_tol=min_tol, args=(H, J1, J2, A1, A2, Ms1, Ms2, ht1, ht2))[0]
                

        # print(phi0)
        # range to solve the equation over
        # print(phi0)
        # if np.isclose(H,0.766, atol=1e-2):
        #     print(phi0)
        ax, aphi, aw, SL_i = continuous_distribution(phi0, H, J1, J2, A1, A2, Ms1, Ms2, ht1, ht2, ret_split=True)
        
        
        pseudo_layers1 = SL_i
        pseudo_layers2 = len(ax) - SL_i
        # print(len(aphi))

        #handle Ms(x) dependance 
        if callable(Ms1):
            Ms1_arr = Ms1(ax[:pseudo_layers1])
        else:
            Ms1_arr = Ms1*np.ones(pseudo_layers1)

        if callable(Ms2):
            Ms2_arr = Ms2(ax[pseudo_layers1:])
        else:
            Ms2_arr = Ms2*np.ones(pseudo_layers2)
        
        # value = 1/(Ms1*len(ax)/2 + Ms2*len(ax)/2)*(Ms1*sum(np.cos(aphi[:int(len(ax)/2)])) + sum(Ms2*np.cos(aphi[int(len(ax)/2):])))
        value = 1/(sum(Ms1_arr) + sum(Ms2_arr))*(sum(Ms1_arr*np.cos(aphi[:pseudo_layers1]))  + sum(Ms2_arr*np.cos(aphi[-pseudo_layers2:])))
        # value = (sum(np.cos(aphi))/len(ax))
        mag[index] = value
        
        
        if sorted:
            upper_bound = phi0 + 0.01
            
            
        if 1 - value < 1e-3 and 1 - mag[index-1] < 1e-3:
            # if we reached saturation no reason to continue computing
            mag[index:] = 1
            break
    
    return mag