# created by Kerry Back, July 2024

__metaclass__ = type

from scipy.stats import norm
from scipy.optimize import root_scalar as root
import pandas as pd 
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import multivariate_normal as binorm

def Gamma(x, y, rho):
    cov = np.array([[1, rho], [rho, 1]]) 
    return binorm.cdf(np.array([x, y]), cov=cov)

phi = lambda z: norm.pdf(z)
Phi = lambda z: norm.cdf(z)

def derivative (fn, x, eps=1.0e-3):
    d1 = max(0, x-eps)
    d2 = min(x+eps, 1)

    return (fn(d2) - fn(d1)) / (d2-d1)

def f(z):
    num = z
    den = z - z*Phi(z) - phi(z)
    return num / den

def finv(t):
    def fns(z):
        num = z
        den = z - z*Phi(z) - phi(z)
        fn = num - t*den
        first = 1 -t*(1 - Phi(z))
        second = t*phi(z) 
        return fn, first, second
    return root(fns, x0=-1, fprime=True, fprime2=True, method='newton').root

def fprime(z):
    num = z
    den = z - z*Phi(z) - phi(z)
    dnum = 1
    dden = 1 - Phi(z)
    return (den*dnum - num*dden) / den**2
    
def g(z, t, xi):
    num = z*Phi(xi) + phi(xi) - z*Phi(z) - phi(z)
    num = z + t*num
    den = z - z*Phi(z) - phi(z)
    return num / den

def ginv(u, t, xi):
    def fns(z) :
        num = z*Phi(xi) + phi(xi) - z*Phi(z) - phi(z)
        num = z + t*num
        den = z - z*Phi(z) - phi(z)
        fn = num - u*den
        first = 1 + t*(Phi(xi) - Phi(z)) -u*(1 - Phi(z))
        second = -t*phi(z) + u*phi(z)
        return  fn, first, second
    a = t*norm.pdf(xi) / (1-t*norm.cdf(xi))
    if u > t:
        return root(fns, x0=(a+finv(t))/2, fprime=True, fprime2=True, method='newton').root
    elif u == t:
        return - t*phi(xi) / (1 - t + t*Phi(xi))
    else:
        print("wrong value for u in ginv", t, u)
        return None

def gprime(z, t, xi):
    num = z*Phi(xi) + phi(xi) - z*Phi(z) - phi(z)
    den = z - z*Phi(z) - phi(z)
    dnum = Phi(xi) - Phi(z)
    dden = 1 - Phi(z)
    return fprime(z) + t * (den*dnum - num*dden) / den**2

# x1 = mu - sigma*e1
# x1 < B iff e1 > -b
# analogous under risk-neutral probability

# E[e1 | e1 > -b]
def truncated_mean(b):
    return phi(b) / Phi(b) 

# E[e1^2 | e1 > -b]
def truncated_square(b):
    return 1 - b * phi(b) / Phi(b)

# e1 and e2 are standard normals with correlation rho
# e1 = rho*e2 + epsilon, so
# E[e1 | e2 > -b] = rho*E[e2 | e2 > -b]
# E[e1**2 | e2 > -b] = rho**2 * E[e2**2 | e2 > -b]

# E[e1 | e1 > -b]
def truncated_mean(b):
    return phi(b) / Phi(b) 

# E[e1^2 | e1 > -b]
def truncated_square(b):
    return 1 - b * phi(b) / Phi(b)

# E[e1 | e1>-b, e2>-b]
def rosenbaum_mean(b, rho):
    return (1+rho) * phi(b) * Phi(b*np.sqrt((1-rho)/(1+rho))) / Gamma(b, b, rho)

# E[e1**2 | e1>-b, e2>-b]
def rosenbaum_square(b, rho):
    term1 = - (1+rho**2) * b * phi(b) * Phi(b*np.sqrt((1-rho)/(1+rho)))
    term2 = rho * np.sqrt( (1-rho**2) / (2*np.pi) ) * phi(  b * np.sqrt( 2/(1+rho) )  )
    return 1 + (term1 + term2) / Gamma(b, b, rho)

# E[e1*e2 | e1>-b, e2>-b]
def rosenbaum_cross(b, rho):
    term1 = - 2 * rho * b * phi(b) * Phi(b*np.sqrt((1-rho)/(1+rho)))
    term2 = np.sqrt((1-rho**2)/(2*np.pi)) * phi(b*np.sqrt(2/(1+rho)))
    return rho + (term1 + term2) / Gamma(b, b, rho)

class Parameters:
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):
        self.mu = mu
        self.mustar = mustar
        self.sigma = sigma
        self.rho = rho 
        self.lam = lam 
        self.gamma = lam * rsquared / (1 + rho) 
        self.delta = (lam-2*self.gamma)*mu
        self.kappa = (mu-mustar) / (self.gamma*(1+rho)*sigma**2)
        self.delta = (lam-2*self.gamma)*mu
        self.sigz = self.gamma * sigma * np.sqrt(1+rho) * np.sqrt((1+rho)/rsquared - 2)
 
def kz(price, mkt, window):
    numdays, numsims = price.shape
    numdays -= 1            # numdays is 1 fewer than number of time grid points

    # daily returns
    r1 = price.pct_change()
    mret = mkt.pct_change()

    # std dev of daily returns
    s = r1.stack().std()
        
    # just to initialize concatenation in a loop below
    # we will delete these later
    r1c = np.zeros((2*window+1,1)) 
    m1c = np.zeros((2*window+1,1))
 
    for sim in range(numsims):

        # 3 sigma returns
        dates = r1[r1[sim].abs()>3*s].index

        dates = [d for d in dates if d>window and d<numdays-window]
        for d in dates:

            # concatenating column vectors
            r1c = np.concatenate(
                (
                    r1c, 
                    # window before, 3 sigma day, and window after
                    r1[sim].loc[d-window: d+window].to_numpy().reshape(-1, 1)
                ),
                axis=1
            )
  
            # contemporaneous market returns
            m1c = np.concatenate(
                (
                    m1c, 
                    mret[sim].loc[d-window: d+window].to_numpy().reshape(-1, 1)
                ),
                axis=1
            )
            
    # convert to dataframes and drop initializing vectors of zeros
    r1c = pd.DataFrame(r1c[:, 1:])
    r1c.index = r1c.index.astype(int) - 30
    r1c.index.name = "day"

    m1c = pd.DataFrame(m1c[:, 1:])
    m1c.index = m1c.index.astype(int) - 30
    m1c.index.name = "day"

    return r1c, m1c


