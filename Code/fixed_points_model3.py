##################################################################
#
# This computes the equilibrium standardized boundary in Model III
#
# Created by Kerry Back, July 2024
#
##################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as binorm
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root
from core.core import *

DIR = "./fixed_points"

def boundary(t, rho):
    def foc(b):
        d = b*np.sqrt((1-rho)/(1+rho))
        cov = np.array([[1,rho],[rho,1]])
        num = t * (1+rho) * phi(b)* (1 - t + t*Phi(d))
        Gamma = binorm.cdf(np.array([b,b]),cov=cov)
        den = (1 - t + t*Phi(b))**2 + t**2 * (Gamma - Phi(b)**2) 
        lhs = b + num / den
        def integrand(z, xi) :
            return (g(z, t, xi) - t) * norm.pdf((xi - b*np.sqrt(1-rho**2))/rho)
        integral = dblquad(
            integrand, 
            - np.inf, 
            finv(t), 
            lambda xi: xi,
            lambda xi: ginv(t, t, xi)
        )[0]
        rhs = (np.sqrt(1-rho**2) / rho) * integral / (1 - t + t*norm.cdf(d))
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