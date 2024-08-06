##############################################################
#
# Execute this file from the command line as
#
# python 1_BoundaryFixedPointApril2024.py "pathname/directory" 
#
# Argument is directory to read from and write to.
#
###############################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root
from core.core import *

DIR = "./fixed_points"

def boundary(t, rho):
    def foc(b):
        num = t * phi(b)
        den = 1 - t + t*Phi(b)
        lhs = b + num / den
        def integrand(z, xi) :
            return (g(z, t, xi) - t) * norm.pdf((xi - b*np.sqrt(1-rho**2))/rho)
        integral = dblquad(
            integrand, 
            -np.inf, 
            finv(t), 
            lambda xi: xi,
            lambda xi: ginv(t, t, xi)
        )[0]
        rhs = (np.sqrt(1-rho**2) / rho) * integral / (1 - t)
        return lhs - rhs   
    return root(foc, x0=-1, x1=1, method='secant').root

times = np.arange(0.01, 1, 0.01) 
rhos = np.arange(0.05, 1, 0.05)
rhos = [0.01] + list(rhos) + [0.99]
df = pd.DataFrame(dtype=float, index=times, columns=rhos)
for rho in rhos :
    print(rho)
    for t in times :
        df.loc[t, rho] = boundary(t, rho)
df.to_csv(f'{DIR}/model2.csv')