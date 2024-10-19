import numpy as np
from scipy.optimize import brentq, minimize
from util import check_callable
from time import time




def extrema(J1,J2,theta1):
    '''function to return the extrema of the boundary condition as a function of theta2'''
    root1 = np.arccos((-J1+np.sqrt(J1**2 + 32*J2**2))/(8*J2)) - theta1
    root2 = -np.arccos((-J1+np.sqrt(J1**2 + 32*J2**2))/(8*J2)) - theta1
    roots = np.array([root1, root2])
    
    if J1 < 2*J2:
        root3 = np.arccos((-J1-np.sqrt(J1**2 + 32*J2**2))/(8*J2)) - theta1
        root4 = -np.arccos((-J1-np.sqrt(J1**2 + 32*J2**2))/(8*J2)) - theta1
        roots = np.array([root1, root2, root3, root4])

    for i in range(len(roots)):
        while roots[i] < 0:
            roots[i] += 2*np.pi
    
    return np.sort(roots)

def boundary_root_find(f, limits, theta1, prev_theta, H, J1, J2, Aex1, Ms, a, show=False, remove=False):
    '''function to root find the boundary condition at the interface as a function of theta2, should stop extrema outside
    of the limits'''

    locals_x = extrema(J1,J2,theta1)
    locals_x = np.sort(locals_x)
    #remove extrema outside the limits
    
    
    locals_x = locals_x[(limits[0] < locals_x) & (limits[1] > locals_x)]
    
    args = (theta1, prev_theta, H, J1, J2, Aex1, Ms, a)
    rtol = 1e-7
    if len(locals_x) == 0:
        
        root = brentq(f,
                    limits[0],
                    limits[1],
                    rtol=rtol,
                    args=args)
        return np.array([root])
    
    
    roots = []
    
    # check from the lower limit to the left most extrema
    # print(f(0, *args))
    if f(limits[0], *args)*f(locals_x[0], *args) < 0:
        
        root = brentq(f,
                    limits[0],
                    locals_x[0],
                    rtol=rtol,
                    args=args)
        roots.append(root)
    
    
    
    # check in between extrema
    # print(f"before entering = {locals_x[1:]}")
    for i in range(1,len(locals_x)):
        
        lower = locals_x[i-1]
        upper = locals_x[i]
        if lower < limits[0] or upper > limits[1]:
            continue

        if f(lower, *args)*f(upper, *args) < 0:
            root  = brentq(f,
                        lower,
                        upper,
                        rtol=rtol,
                        args=args)
            roots.append(root)
            
    # check from the right most extreme to the upper limit
    
    if f(limits[1], *args)*f(locals_x[-1], *args) < 0: 
        root = brentq(f,
                    locals_x[-1],
                    limits[1],
                    rtol=rtol,
                    args=args)
        roots.append(root)
        
        
        
    return np.array(roots)