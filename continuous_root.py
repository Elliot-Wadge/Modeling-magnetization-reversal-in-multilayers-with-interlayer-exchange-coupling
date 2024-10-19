import numpy as np
from scipy.optimize import brentq, minimize
from util import check_callable

# replaced by extrema
def boundary_max(J1,J2,c):
    '''return the maximum value of the boundary condition as a function of J1, J2, and c'''
    
    return np.arccos((-J1 + np.sqrt(J1**2 + 32*J2**2))/(8*J2)) - c


def extrema(J1,J2,theta1):
    '''function to return the extrema of the boundary condition at the spacer layer as a function of theta2
    this facilitates faster root finding
    
    :param float J1: the bilinear coupling in milli-J/m^2
    :param float J2: the biquadratic coupling in milli-J/m^2
    :param float theta1: the angle on the known side of spacer layer
    '''
    
    
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


def root_find(f, limits, theta1, w1, J1, J2, Aex1, show=False, remove=False):
    '''function to root find the boundary condition at the spacerlayer as a function of theta2, should stop extrema outside
    of the limits, return the roots, there may be multiple. I think there should be a way to narrow it down to one but I haven't figured it out yet'''

    # first find the extrema of the boundary condition, the zero solution will be in between extrema
    locals_x = extrema(J1,J2,theta1)
    locals_x = np.sort(locals_x)
    
    
    #remove extrema outside the limits
    locals_x = locals_x[(limits[0] < locals_x) & (limits[1] > locals_x)]
    
    args = (theta1, w1, J1, J2, Aex1)
    
    
    # first handle the case where all extrema are outside of the limites
    if len(locals_x) == 0:
        
        root = brentq(f,
                    limits[0],
                    limits[1],
                    xtol=1e-4,
                    args=args)
        return np.array([root])
    
    
    roots = []
    
    # check from the lower limit to the left most extrema
    # print(f(0, *args))
    if f(limits[0], *args)*f(locals_x[0], *args) < 0:
        
        root = brentq(f,
                    limits[0],
                    locals_x[0],
                    xtol=1e-4,
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
                        xtol=1e-4,
                        args=args)
            roots.append(root)
            
    # check from the right most extreme to the upper limit
    if f(limits[1], *args)*f(locals_x[-1], *args) < 0: 
        root = brentq(f,
                    locals_x[-1],
                    limits[1],
                    xtol=1e-4,
                    args=args)
        roots.append(root)
        
        
        
    return np.array(roots)



def get_A_transitions(A, half_thickness):
    '''find where the stiffness changes if anywhere inside the bulk, return the positions at which a transition occurs
    
    :param A: the exchange stiffness in the bulk to be propegated 10^-11 J/m
    :type A: float or callable,
    :param float half_thickness: thickness of one side of device (seperated in two by spacer layer)'''
    
    x = np.linspace(0, half_thickness, half_thickness*100)
    
    A_arr = check_callable(A,x)
    
    if isinstance(A_arr, np.ndarray):
        idx = np.where(A_arr[:-1] != A_arr[1:])[0]
    else:
        idx = []
    
    
    if len(idx) == 0:
        transitions = [half_thickness]
    else:
        transitions = np.ones(len(idx) + 1)
        transitions[:len(idx)] = x[idx]
        transitions[-1] = half_thickness

    return transitions

def non_zero(f, a, b, step, args=[]):
    '''function to help brute force the root finding, finds the first non-zero
    value between a and b by taking steps of size step'''
    total = b-a
    c = a
    
    steps = 0
    
    while total > steps:
        steps += step
        y_c = f(c,*args)
        # print(c)
        if y_c != 0:
            return c
        
        c += step
    
    return None



def non_zero_edge(f,a,b,tol,args=[]):
    '''binary search for finding the boundary of a non-zero region'''
    
    if abs(a-b) < tol:
        return a
    
    c = (a+b)/2
    i = 0
    while abs(a-b) > tol:
        i += 1
        y_c = f(c,*args)
        y_a = f(a,*args)
        if np.isclose(y_c, 0, atol=1e-11):
            
            b = c
            c = (a+c)/2
            
        elif y_c != 0:
            
            a = c
            c = (b+c)/2
            
    return a

def find_edges(f, center, outer1, outer2, args=[], tol=1e-5):
    '''function to help brute force root find while maintaining speeds,
    the root finding is the hardest part of the problem'''
    edge1 = non_zero_edge(f, center, outer1, tol=tol, args=args)
    edge2 = non_zero_edge(f, center, outer2, args=args, tol=tol)
    return edge1, edge2


def root_v2(f, a, b, tol, args=[], step=1e-2, min_tol=1e-5, show=False):
    '''function to brute force find the solution to the boundary condition as quickly as possible,
    this is the main bottleneck'''
    
    c = None
    max_iter = 1
    i = 0
    while c is None and i <= max_iter:
        c = non_zero(f, a, b, step, args=args)
        step /= 10
        i += 1
    e = find_edges(f, c, a, b, tol=tol, args=args)


    # e = (a, b)
    if show:
        print(c,e)
    
    if False:
        
        root = brentq(f, e[0], e[1], rtol=tol,args=args)
    else:
        new_f = lambda x: abs(f(x, *args))
        # root = 1
        # while root > 1e-7
        found_min = 1
        min_root = 0
        guess = e[1]
        max_iter = 5
        count = 0
        
        while found_min > min_tol:
            
            root = minimize(new_f, (guess), tol=1e-9, bounds=((e[0],e[1]),), method="L-BFGS-B").x
            if found_min > new_f(root):
                found_min = new_f(root)
                min_root = root
                
            if show:
                print(f"guess={guess}, new_min={new_f(root)}, found_min={found_min}, {found_min > new_f(root)}")
                
            guess = (3*guess + e[0])/4 #weighted average
            
            if count > max_iter:
                break
                #return min_root
            
            count += 1
        # root = quick_brute(f, e[0], e[1], tol=tol, args=args)[0]
    return min_root
