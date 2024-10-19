import numpy as np
from util import arrarize, H_of_0
from d_torque_root import boundary_root_find
from continuous_root import root_v2

def update_rule(H, theta, prev_theta, A1, A2, a1, a2, Ms):
    '''discrete update rule from the torque model
    
    :param float H: magnetic field strength in tesla 
    :param float theta: theta of known current magnetic moment in radians
    :param float prev_theta: angle of the magnetic moment before current in radians
    :param float A1: exchange stiffness between current layer and previous layer 10^-11 J/m
    :param float A2: exchange stiffness between current layer and next layer 10^-11 J/m
    :param float a1: spacing between current layer and previous layer in 10^-8 m
    :param float a2: spacing between current layer and next layer in 10^-8 m
    :param float Ms: magnitude of magnetic moment of current layer mega-A/m'''
    
    if abs((a2*A1/(a1*A2))*np.sin(theta-prev_theta) + Ms*H*a1*a2*np.sin(theta)/(2*A2)) > 1:
        # print(f"update rule => {(a2*A1/(a1*A2))*np.sin(theta-prev_theta) + Ms*H*a1*a2*np.sin(theta)/(2*A2)}")
        # print(f"H = {H}, a = {theta-prev_theta}, b = {theta}")
        return theta + np.pi/2 
    ret = theta + np.arcsin((a2*A1/(a1*A2))*np.sin(theta-prev_theta) + Ms*H*a1*a2*np.sin(theta)/(2*A2))
    # if ret > np.pi:
    #     ret = np.pi
    return ret


def first_point(theta, H, A, a, Ms):
    '''solution to the boundary condition at the first edge, returns the theta second from edge
    
    :param float theta: outermost angle in radian
    :param float H: external field strength in tesla
    :param float A: exchange stiffness between outermost and next angle in 10^-11 J/m
    :param float a1: spacing between outermost and next angle in 10^-8 m
    :param float Ms: magnitude of magnetic moment of outermost layer'''
    
    A *= 100 #unit conversion
    Ms *= 0.1 #unit conversion
    next_theta = theta + np.arcsin(a**2*Ms*H/(2*A) * np.sin(theta))
    if abs(a**2*Ms*H/(2*A) * np.sin(theta)) > 1:
        print(f"first point => {a**2*Ms*H/(2*A) * np.sin(theta)}")

    return next_theta

def propagate(layers, H, prev_theta, theta, A, a, Ms):
    '''propagate the torque model for N layers and return the resulting array of angles
    
    :param int layers: number of layers to be propagated 
    :param float H: external field strength in tesla
    :param float prev_theta: angle of layer one before current layer
    :param float theta: angle of current layer
    :param A: exchange stiffness in the bulk 10^-11 J/m
    :type A: float or np.ndarray
    :param float a: spacing in the bulk 10^-8 m
    :type a: float or np.ndarray
    :param Ms: magnitude of magnetic moment in the bulk
    :type Ms: float or np.ndarray
    '''
    
    A_arr = arrarize(A, layers-1)*100
    a_arr = arrarize(a, layers-1)
    Ms_arr = arrarize(Ms, layers)*0.1
    
    # ret = [theta, next_theta]
    ret = np.zeros(layers)
    ret[0] = prev_theta
    ret[1] = theta

    
    for i in range(1,layers-1):
        
        new_theta = update_rule(H, theta, prev_theta, A_arr[i-1], A_arr[i], a_arr[i-1], a_arr[i], Ms_arr[i])
        prev_theta = theta
        theta = new_theta
        
        ret[i+1] = new_theta

    return np.array(ret)


def boundary(next_theta, theta, prev_theta, H, J1, J2, A, Ms, a):
    '''boundary condition at the spacer layer
     
    :param float next_theta: angle on opposite side of spacer in radians
    :param float theta: angle before spacer in radians
    :param float prev_theta: angle two before spacer in radians
    :param float H: external field strength in tesla
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A: exchange stiffness between prev_theta and theta in 10^-11 J/m
    :type A: float or np.ndarray
    :param float a: spacing between prev_theta and theta in 10^-8 m
    :type a: float or np.ndarray
    :param Ms: magnitude of magnetic moment of theta
    :type Ms: float or np.ndarray'''
    
    
    A *= 100 #unit conversion
    Ms *= 0.1 #unit conversion
    return 2*A/a * np.sin(theta-prev_theta) - J1*np.sin(theta + next_theta) - J2*np.sin(2*(theta + next_theta)) + a*Ms*H*np.sin(theta)

def edge_boundary(thetad, thetad_1, H, A, a, Ms, show = False):
    '''boundary condition at the edge to be check after propagating the solution'''
    
    
    A *= 100 #unit conversion
    Ms *= 0.1 #unit conversion
    if show:
        print(f"thetad  = {thetad}\nthetad_1 = {thetad_1}\nA = {A}\na = {a}\nMs = {Ms}")
    
    ret = 2*A/a*np.sin(thetad - thetad_1) + a*Ms*H*np.sin(thetad)
    
    
    return ret


def torque_distribution(theta0, H, J1, J2, A1, A2, N1, N2, Ms1, Ms2, a1, a2):
    '''take an initial starting angle theta and system parameters and return the propagated distribution
    
    :param float theta0: starting at angle at outer edge in radians
    :param float H: external field strength in tesla
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A: exchange stiffness in the bulk 10^-11 J/m
    :type A: float or np.ndarray
    :param float a: spacing in the bulk 10^-8 m
    :type a: float or np.ndarray
    :param Ms: magnitude of magnetic moment in the bulk
    :type Ms: float or np.ndarray'''
    
    A1 = arrarize(A1, N1-1)
    a1 = arrarize(a1, N1-1)
    Ms1 = arrarize(Ms1, N1)
    A2 = arrarize(A2, N2-1)
    a2 = arrarize(a2, N2-1)
    Ms2 = arrarize(Ms2, N2)
    
    

    #propagate to interface
    theta1_1 = first_point(theta0, H, A1[0], a1[0], Ms1[0])
    theta1 = propagate(N1, H, theta0, theta1_1, A1, a1, Ms1)
    
    # same as in the continuous model, they share code here so something to be careful of
    roots = boundary_root_find(boundary, (0, np.pi-theta1[-1]), theta1[-1], theta1[-2], H, J1, J2, A1[-1], Ms1[-1], a1[-1])



    index = None
    min_theta2d = np.inf
    for i,root in enumerate(roots):
        #integers in this equation are for unit conversions A *= 100 Ms *= 0.1

        theta2_2_temp = root - np.arcsin(A1[-1]/A2[0] * a2[0]/a1[-1] * np.sin(theta1[-1] - theta1[-2]) + a2[0]*H/(100*2*A2[0])*(a1[-1]*Ms1[-1]*0.1*np.sin(theta1[-1]) - a2[0]*Ms2[0]*0.1*np.sin(root)))
        
        if abs(A1[-1]/A2[0] * a2[0]/a1[-1] * np.sin(theta1[-1] - theta1[-2])) > 1:
            print(f"theta2_2_temp => {A1[-1]/A2[0] * a2[0]/a1[-1] * np.sin(theta1[-1] - theta1[-2])}")
        
        sol2_temp = propagate(N2, H, root, theta2_2_temp, A2, a2, Ms2)
        
        # if i == 0  and not np.any(theta2_2_temp > np.pi) and not np.any(theta2_2_temp < 0):
        #     #this is not correct
        #     # abs(edge_boundary(sol2_temp[-1], sol2_temp[-2], H, A2, a2, Ms2))
        #     min_theta2d = abs(edge_boundary(sol2_temp[-1], sol2_temp[-2], H, A2[-1], a2[-1], Ms2[-1]))
        #     theta2_2 = theta2_2_temp
        #     index = 0

        if abs(edge_boundary(sol2_temp[-1], sol2_temp[-2], H, A2[-1], a2[-1], Ms2[-1])) < min_theta2d and np.all(sol2_temp <= np.pi) and np.all(sol2_temp >= 0):
            
            index = i
            theta2_2 = theta2_2_temp
            min_theta2d = abs(edge_boundary(sol2_temp[-1], sol2_temp[-2], H, A2[-1], a2[-1], Ms2[-1]))
    
    
    
    if index is not None:
        theta2 = propagate(N2, H, roots[index], theta2_2, A2, a2, Ms2)
    else:
        raise(ValueError("initial angle could not be propagated"))
    
    return np.concatenate([theta1, theta2])


def torque_shoot(theta0, H, J1, J2, A1, A2, N1, N2, Ms1, Ms2, a1, a2, show=False):
    
    '''wrapper for A_shoot_distribution that returns the final boundary condition instead of the distribution
    
    :param float theta0: starting at angle at outer edge in radians
    :param float H: external field strength in tesla
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A1: exchange stiffness in the bulk1 10^-11 J/m
    :type A1: float or np.ndarray
    :param A2: exchange stiffness in the bulk2 10^-11 J/m
    :type A2: float or np.ndarray
    :param int N1: number of layers on side 1 of spacer layer
    :param int N2: number of layers on side 2 of spacer layer
    :param Ms1: magnitude of magnetic moment in the bulk1
    :type Ms1: float or np.ndarray
    :param Ms2: magnitude of magnetic moment in the bulk2
    :type Ms2: float or np.ndarray
    :param float a1: spacing in the bulk1 10^-8 m
    :type a1: float or np.ndarray
    :param float a2: spacing in the bulk2 10^-8 m
    :type a2: float or np.ndarray
    '''
    
    A1 = arrarize(A1, N1-1)
    a1 = arrarize(a1, N1-1)
    Ms1 = arrarize(Ms1, N1)
    A2 = arrarize(A2, N2-1)
    a2 = arrarize(a2, N2-1)
    Ms2 = arrarize(Ms2, N2)
    try:
        theta = torque_distribution(theta0, H, J1, J2, A1, A2, N1, N2, Ms1, Ms2, a1, a2)
        if show:
            print(f"final condition {Ms2[-1]}")
        return edge_boundary(theta[-1], theta[-2], H, A2[-1], a2[-1], Ms2[-1], show=show)
    except:
        return 0
        # return 1/2*(theta0-0.4)**2 + 0.2
        # return abs((theta0-np.pi/4)+1)
        # return 1


def torque_M(field, J1, J2, A1, A2, N1, N2, Ms1, Ms2, a1, a2, tol=1e-3, step=1e-2, min_tol=1e-5, sorted=True):
    
    '''function to return M as a function of H and system parameters
    
    :param np.ndarray: field strengths at which to calculate M(H)
    :param float H: external field strength in tesla
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param A1: exchange stiffness in the bulk1 10^-11 J/m
    :type A1: float or np.ndarray
    :param A2: exchange stiffness in the bulk2 10^-11 J/m
    :type A2: float or np.ndarray
    :param int N1: number of layers on side 1 of spacer layer
    :param int N2: number of layers on side 2 of spacer layer
    :param Ms1: magnitude of magnetic moment in the bulk1
    :type Ms1: float or np.ndarray
    :param Ms2: magnitude of magnetic moment in the bulk2
    :type Ms2: float or np.ndarray
    :param float a1: spacing in the bulk1 10^-8 m
    :type a1: float or np.ndarray
    :param float a2: spacing in the bulk2 10^-8 m
    :type a2: float or np.ndarray
    '''
    
    
    ret = np.ones(len(field))
    for i,H in enumerate(field):

        
            

        args = (H, J1, J2, A1, A2, N1, N2, Ms1, Ms2, a1, a2)
        upper_bound = np.pi

        try:

            if H == 0:
                theta0 = H_of_0(J1, J2, Ms1, Ms2)
                # right now has a problem with theta0 = np.pi
                
            else:
                # theta0 = quick_brute(A_shoot, 0, upper_bound+2*tol, tol, args=args)[0]
                
                theta0 = root_v2(torque_shoot, 1e-3, upper_bound+1e-3, tol=tol, step=step, args=args, min_tol=min_tol)
                
            # if H < 0.02:
            #     print(f"d torque theta0 = {theta0}")
            
            theta = torque_distribution(theta0, *args)
        except:
            
            theta =np.zeros(N1 + N2)
        

        if hasattr(Ms1, "__iter__"):
            Ms1_sum = np.sum(Ms1)
        else:
            Ms1_sum = N1*Ms1

        if hasattr(Ms2, "__iter__"):
            Ms2_sum = np.sum(Ms2)
        else:
            Ms2_sum = N2*Ms2


        mag = 1/(Ms1_sum + Ms2_sum) * (np.sum(Ms1*np.cos(theta[:N1])) + np.sum(Ms2*np.cos(theta[N1:])))
        ret[i] = mag
        # if mag < 0:
        #     print(theta0)
            
        if sorted:
            upper_bound = theta[0]


        if abs(1-mag) < 0.002:
            break

        


    return abs(ret)