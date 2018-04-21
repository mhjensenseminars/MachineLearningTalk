import numpy as np
from scipy.optimize import minimize
def DerivativeE(x):
    return x[0]-1.0/(4*x[0]*x[0]*x[0]);

def Energy(x):
   return x[0]*x[0]*0.5+1.0/(8*x[0]*x[0]);
x0 = np.zeros(1)
x0 = 1.0
res = minimize(Energy, x0, method='powell', options={'xtol': 1e-8, 'disp': True})

#res = minimize(Energy, x0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
#res = minimize(rosen, x0, method='nelder-mead',