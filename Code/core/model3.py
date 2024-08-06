# created by Kerry Back, July 2024

__metaclass__ = type

from scipy.stats import norm
import pandas as pd 
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from scipy.optimize import root_scalar as root
from core.core import *

class Stage2(Parameters):
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):
        super(Stage2, self).__init__(mu, mustar, sigma, rho, lam, rsquared)
    
    # assuming firm 2 reveals x2 at time t when the boundary is b
    def price(self, t, b, x2, u):
        mustar1 = self.rho * x2 + (1-self.rho) * self.mustar
        sig1 = self.sigma*np.sqrt(1-self.rho**2)
        xi = (b - mustar1) / sig1
        if f(xi) <= u:
            return mustar1 + sig1*finv(u)
        else :
            return mustar1 + sig1*ginv(u, t, xi)

    def rn_probs(self, t, b, x2, u):
        mustar1 = self.rho * x2 + (1-self.rho) * self.mustar
        sig1 = self.sigma*np.sqrt(1-self.rho**2)
        xi = (b - mustar1) / sig1
        p = self.price(t, b, x2, u)
        z = (p - mustar1) / sig1
        q1 = t * Phi(xi)
        q2 = (u-t) * Phi(z)
        q3 = 1 - u
        sum = q1 + q2 + q3 
        return q1 / sum, q2 / sum, q3 / sum

    def price_schedule(self, t, b, x2, u):
        mustar1 = self.rho * x2 + (1-self.rho) * self.mustar
        sig1 = self.sigma*np.sqrt(1-self.rho**2)
        xi = (b - mustar1) / sig1
        q1, q2, q3 = self.rn_probs(t, b, x2, u)
        p = self.price(t, b, x2, u)
        z = (p - mustar1) / sig1
        zdot = 1 / gprime(z, t, xi) if p > b else 1 / fprime(z)
        p2 = - phi(z) * zdot / Phi(z)
        p1 = 0 if p > b else p2
        p3 = Phi(-z) / (1-t)
        e1 = e2 = p
        e3 = mustar1 + sig1 * phi(z) / Phi(-z)
        num = (p1 * e1 * q1) + (p2 * e2 * q2) + (p3 * e3 * q3)
        den = (p1 * q1) + (p2 * q2) + (p3 * q3)
        return num / den

    def time(self, t, b, x2, x1):

        mustar1 = self.rho * x2 + (1-self.rho) * self.mustar
        sig1 = self.sigma*np.sqrt(1-self.rho**2)
        xi = (b - mustar1) / sig1
        z = (x1 - mustar1) / sig1
        u1 = max(t, f(z))
        u2 = max(t, g(z, t, xi))
        return u2 if x1 >= b else u1

def Eq42(b, t, rho):
    d = b*np.sqrt((1-rho)/(1+rho))
    cov = np.array([[1,rho],[rho,1]])
    num = t * (1+rho) * phi(b)* (1 - t + t*Phi(d))
    Gamma = multivariate_normal.cdf(np.array([b,b]),cov=cov)
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
    return lhs, rhs   

def Solve(t, rho):
    def foc(b):
        lhs, rhs = Eq42(b, t, rho)
        return lhs - rhs   
    return root(foc, x0=-1, x1=1, method='secant').root

fixed_points = pd.read_csv('fixed_points/model3.csv',index_col="time")
fixed_points.columns = np.round(fixed_points.columns.astype(float), 2)

class Stage1(Parameters):
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):

        super(Stage1, self).__init__(mu, mustar, sigma, rho, lam, rsquared)

        if rho in fixed_points.columns:
            boundaries = [mustar + sigma*float(b) for b in fixed_points[rho]]
            times = [float(t) for t in fixed_points.index]
        else:
            times = np.linspace(1.0e-3, 1-1.0e-3, 21)
            boundaries = [mustar + sigma*Solve(t, rho) for t in times]

        self.from_t_to_b = splrep(times, boundaries)

        # input to splrep must be ordered from smallest to largest
        self.from_b_to_t = splrep(boundaries[::-1], times[::-1])

    def bdy(self, t):
        try:
            # should work for a numpy array of times
            return pd.Series(splev(t, self.from_t_to_b, der=0), index=t)
        except:
            # should work for an individual time
            assert t>=0 and t<=1, "time must be between 0 and 1"
            return splev(t, self.from_t_to_b, der=0).item()
            
    def time(self, b):
        # for an individual boundary point b
        if b >= self.bdy(0):
            return 0
        elif b <= self.bdy(1):
            return 1
        else:
            return min(1., max(splev(b, self.from_b_to_t, der=0).item(), 0.))
    
    # conditional probs of four events
    def probs(self, t):
        b = (self.bdy(t) - self.mu) / self.sigma
        q1 = (1-t)**2                        # neither informed
        q2 = t * (1-t) * Phi(b)              # 1 informed 
        q3 = t * (1-t) * Phi(b)              # 2 informed
        q4 = t**2 * Gamma(b, b, self.rho)    # both informed
        total = q1 + q2 + q3 + q4 
        return q1/total, q2/total, q3/total, q4/total 

    # risk-neutral conditional probs of four events
    def rn_probs(self, t):
        b = (self.bdy(t) - self.mustar) / self.sigma
        q1 = (1-t)**2                        # neither informed
        q2 = t * (1-t) * Phi(b)              # 1 informed 
        q3 = t * (1-t) * Phi(b)              # 2 informed
        q4 = t**2 * Gamma(b, b, self.rho)    # both informed
        total = q1 + q2 + q3 + q4 
        return q1/total, q2/total, q3/total, q4/total 

    # price: x = mu* - sigma*e*, so E*[x] = mu* - sigma*E*[e*]
    def price(self, t):
        q1, q2, q3, q4 = self.rn_probs(t)
        b = (self.bdy(t) - self.mustar) / self.sigma
        a1 = 0
        a2 = truncated_mean(b)
        a3 = self.rho * truncated_mean(b)
        a4 = rosenbaum_mean(b, self.rho)
        return self.mustar - self.sigma * (q1*a1 + q2*a2 + q3*a3 + q4*a4)

    def price_schedule(self, t):
        q1, q2, q3, q4 = self.rn_probs(t)
        b = self.bdy(t)
        z = (b - self.mustar) / self.sigma
        zdot = derivative(lambda u: self.bdy(u) / self.sigma, t)
        p1 = - phi(z) * zdot * Phi(z*np.sqrt((1-self.rho)/(1+self.rho)))
        p1 /= Gamma(z, z, self.rho)
        p2 = - phi(z) * zdot / Phi(z)
        p3 = Gamma(-z, z, -self.rho) / ((1-t)*Phi(z))
        p4 = Phi(-z) / (1-t)
        e1 = b
        e2 = b
        num = 1 - (1+self.rho)*Phi(-z*np.sqrt((1-self.rho)/(1+self.rho)))
        e3 = self.mustar + self.sigma * phi(z) * num / Gamma(z, -z, -self.rho)
        e4 = self.mustar + self.sigma * phi(z) / Phi(-z)
        num = p1*e1*q1 + p2*e2*q2 + p3*e3*q3 + p4*e4*q4
        den = p1*q1 + p2*q2 + p3*q3 + p4*q4
        return num / den

    # price of w 
    # w = delta* + gamma*(x1+x2) + z* where delta* = delta - kappa*sigma_z**2, 
    # so price of w = delta* + 2*gamma*price
    def price_of_w(self, t):
        return self.delta - self.kappa*self.sigz**2 + 2*self.gamma*self.price(t)

    # other moments are under physical distribution

    def mean_of_e(self, t):
        q1, q2, q3, q4 = self.probs(t)
        b = (self.bdy(t) - self.mu) / self.sigma
        a1 = 0
        a2 = truncated_mean(b)
        a3 = self.rho * truncated_mean(b)
        a4 = rosenbaum_mean(b, self.rho)
        return q1*a1 + q2*a2 + q3*a3 + q4*a4

    def mean_of_e_squared(self, t):
        q1, q2, q3, q4 = self.probs(t)
        b = (self.bdy(t) - self.mu) / self.sigma
        a1 = 1
        a2 = truncated_square(b)
        a3 = self.rho**2 * truncated_square(b) + 1 - self.rho**2
        a4 = rosenbaum_square(b, self.rho)
        return q1*a1 + q2*a2 + q3*a3 + q4*a4

    def mean_of_e1e2(self, t):
        q1, q2, q3, q4 = self.probs(t)
        b = (self.bdy(t) - self.mu) / self.sigma
        a1 = self.rho 
        a2 = self.rho * truncated_square(b)
        a3 = a2 
        a4 = rosenbaum_cross(b, self.rho)
        return q1*a1 + q2*a2 + q3*a3 + q4*a4


    # physical mean: x = mu - sigma*e, so E[x] = mu - sigma*E[e]
    def mean(self, t):
        return self.mu - self.sigma * self.mean_of_e(t)
        
    # mean of x**2: x = mu - sigma*e, so
    # E[x**2] = mu**2 - 2*mu*sigma*E[e] + sigma**2 * E[e**2]
    def mean_of_square(self, t):
        return (
            self.mu**2 
            - 2 * self.mu * self.sigma * self.mean_of_e(t)
            + self.sigma**2 * self.mean_of_e_squared(t)
        )
    
    # mean of x1 * x2
    # x1*x2 = (mu-sigma*e1)*(mu-sigma*e2) 
    #       = mu**2 - mu*sigma*(e1+e2) + sigma**2 * e1*e2
    def mean_of_x1x2(self, t):
        return (
            self.mu**2 
            - 2 * self.mu * self.sigma * self.mean_of_e(t)
            + self.sigma**2 * self.mean_of_e1e2(t)
        )

    # mean of w: w = delta + gamma*(x1+x2) + z, so E[w] = delta + 2*gamma*E[x]
    def mean_of_w(self, t):
        return self.delta + 2 * self.gamma * self.mean(t)

    

    # mean of w**2
    # w = delta + gamma*(x1+x2) + z, so
    # w**2 = delta**2 + gamma**2 *(x1**2 + x2**2 + 2*x1*x2) + z**2
    #      + 2 * delta * gamma*(x1+x2) + 2*delta*z + 2*gamma*(x1+x2)*z
    def mean_of_w_squared(self, t):
        return (
            self.delta**2
            + 2 * self.gamma**2 * self.mean_of_square(t)
            + 2 * self.gamma**2 * self.mean_of_x1x2(t)
            + self.sigz**2
            + 4 * self.delta * self.gamma * self.mean(t)
        )

    # mean of x1*w
    # w = delta + gamma*(x1+x2) + z, so
    # x1*w = delta*x1 + gamma*x1**2 + gamma*x1*x2 + x1*z 
    def mean_of_wx(self, t):
        return (
            self.delta * self.mean(t) 
            + self.gamma * self.mean_of_square(t)
            + self.gamma * self.mean_of_x1x2(t)
        )

    def varx(self, t):
        return self.mean_of_square(t) - self.mean(t)**2

    def varw(self, t):
        return self.mean_of_w_squared(t) - self.mean_of_w(t)**2

    def covwx(self, t):
        return self.mean_of_wx(t) - self.mean(t) * self.mean_of_w(t) 

    def beta(self, t):
        return self.covwx(t) / self.varw(t) 

    def alpha(self, t):
        return (
            self.mean(t) - self.price(t) 
            - self.beta(t) * (self.mean_of_w(t) - self.price_of_w(t))
        )

    def alpha_wrong(self, t):
        return (
            self.mean(t) - self.price(t) 
            - self.beta(0) * (self.mean_of_w(t) - self.price_of_w(t))
        )

class DailySim(Parameters):
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):
        super(DailySim, self).__init__(mu, mustar, sigma, rho, lam, rsquared)
        self.stage1 = Stage1(mu, mustar, sigma, rho, lam, rsquared)
        self.stage2 = Stage2(mu, mustar, sigma, rho, lam, rsquared)
    def theta(self, numsims):
        return np.random.uniform(low=0, high=1 - 1.0e-6, size=(numsims, 2))
    def x(self, numsims):
        corr = np.array([[1, self.rho], [self.rho, 1]])
        cov = self.sigma**2 * corr
        return multivariate_normal.rvs([self.mu, self.mu], cov, size=numsims)
    def sim(self, numdays, numsims):
        theta = self.theta(numsims)
        x = self.x(numsims)
        grid = np.linspace(0, 0.9999, numdays + 1)
        grid = np.round(grid, 3)
        bdy = self.stage1.bdy(grid)
        price = pd.Series([self.stage1.price(t) for t in grid], index=grid)
        sim_price1 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        sim_price2 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        for sim in range(numsims):
            theta1 = theta[sim, 0]
            theta2 = theta[sim, 1]
            x1 = x[sim, 0]
            x2 = x[sim, 1]
            tau1 = self.stage1.time(x1)
            tau2 = self.stage1.time(x2)
            tau = min(theta2, max(theta1, tau1))

            # pre first announcement

            # last date in grid before first announcement
            tau_in_grid = np.max([t for t in grid if t < tau])
            sim_price1.loc[:tau_in_grid, sim] = price.loc[:tau_in_grid]
            sim_price2.loc[:tau_in_grid, sim] = price.loc[:tau_in_grid]

            # first date in grid after first announcement
            tau_in_grid = np.min([t for t in grid if t >= tau]) 

            # firm 1 discloses first
            if max(theta1, tau1) < theta2:
        
                # firm 1's price is x1 after disclosing
                sim_price1.loc[tau_in_grid:, sim] = x1

                # firm 2's price is second stage and then x2
                t = max(theta1, tau1)      # firm 1's disclosure time
                B = self.stage1.bdy(t)          # boundary when firm 1 discloses
                d = np.min(                # first date afte firm 2 discloses
                    [
                        u for u in grid 
                        if u >= max(t, theta2)
                        and x2 >= self.stage2.price(t, B, x1, u)
                    ]
                )
                period = [u for u in grid if u>=t and u<d]
                for u in period:               # until firm 2 discloses
                    sim_price2.loc[u, sim] = self.stage2.price(t, B, x1, u)
                sim_price2.loc[d:, sim] = x2   # after firm 2 discloses
       
            # firm 2 discloses first
            else:

                # firm 2's price is x2 after disclosing
                sim_price2.loc[tau_in_grid:, sim] = x2

                # firm 1's price is second stage and then x1
                t = max(theta2, tau2)      # firm 2's disclosure time
                B = self.stage1.bdy(t)          # boundary when firm 2 discloses
                d = np.min(                # first date after firm 1 discloses
                    [
                        u for u in grid 
                        if u >= max(t, theta1)
                        and x1 >= self.stage2.price(t, B, x2, u)
                    ]
                )
                period = [u for u in grid if u>=t and u<d]
                for u in period:               # until firm 1 discloses
                    sim_price1.loc[u, sim] = self.stage2.price(t, B, x2, u)
                sim_price1.loc[d:, sim] = x1   # after firm 1 discloses
     
            if (sim+1) % 1000 == 0:
                print(f"sim {sim+1}") 
        sim_mkt = (
            self.delta 
            - self.kappa * self.sigz**2
            + self.gamma * (sim_price1 + sim_price2)
        )
        theta = pd.DataFrame(theta, columns=["theta1", "theta2"])
        theta.index.name = "sim"
        x = pd.DataFrame(x, columns=["x1", "x2"])
        x.index.name = "sim"
        sim_price1 = sim_price1.reset_index(drop=True)
        sim_price1.index.name = "day"
        sim_price2 = sim_price2.reset_index(drop=True)
        sim_price2.index.name = "day"
        sim_mkt = sim_mkt.reset_index(drop=True)
        sim_mkt.index.name = "day"
        return theta, x, sim_price1, sim_price2, sim_mkt

class Sim(Parameters):
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):
        super(Sim, self).__init__(mu, mustar, sigma, rho, lam, rsquared)
        self.stage1 = Stage1(mu, mustar, sigma, rho, lam, rsquared)
        self.stage2 = Stage2(mu, mustar, sigma, rho, lam, rsquared)

    def theta(self, numsims):
        return np.random.uniform(low=0, high=1 - 1.0e-6, size=(numsims, 2))

    def x(self, numsims):
        corr = np.array([[1, self.rho], [self.rho, 1]])
        cov = self.sigma**2 * corr
        return multivariate_normal.rvs([self.mu, self.mu], cov, size=numsims)
        
    def sim(self, numsims):
        
        df = pd.DataFrame(
            dtype=float, 
            index=range(numsims),
            columns=[
                "theta1", "theta2", "x1", "x2", 
                "tau1", "tau2", "price1", "price2", 
                "price_pre", "price_post",
                "price_schedule1", "price_schedule2"
            ]
        )
        theta = self.theta(numsims)
        x = self.x(numsims)
        df.theta1 = theta[:, 0]
        df.theta2 = theta[:, 1]
        df.x1 = x[:, 0]
        df.x2 = x[:, 1]
       
        for sim in range(numsims):

            if (sim+1) % 1000 == 0: print(sim+1)

            theta1 = theta[sim, 0]
            theta2 = theta[sim, 1]
            x1 = x[sim, 0]
            x2 = x[sim, 1]
            tau1 = max(self.stage1.time(x1), theta1)
            tau2 = max(self.stage1.time(x2), theta2)
         
            # firm 1 discloses first
            if tau1 < tau2:

                # price of firm 1 before announcement
                df.loc[sim, "tau1"] = tau1
                df.loc[sim, "price1"] = self.stage1.price(tau1)

                # price of firm 1 after scheduling
                df.loc[sim, "price_schedule1"] = self.stage1.price_schedule(tau1)

                # price of firm 2 before firm 1 announcement
                df.loc[sim, "price_pre"] = df.loc[sim, "price1"]

                # price of firm 2 following firm 1 announcement
                b = self.stage1.bdy(tau1)
                df.loc[sim, "price_post"] = self.stage2.price(tau1, b, x1, tau1)

                # price of firm 2 before its announcement
                tau2 = max(self.stage2.time(tau1, b, x1, x2), theta2)
                df.loc[sim, "tau2"] = tau2 
                df.loc[sim, "price2"] = self.stage2.price(tau1, b, x1, tau2)
            
                # price of firm 2 after scheduling
                df.loc[sim, "price_schedule2"] = self.stage2.price_schedule(tau1, b, x1, tau2) 
            
            # firm 2 discloses first
            else:

                # price of firm 2 before announcement
                df.loc[sim, "tau2"] = tau2
                df.loc[sim, "price2"] = self.stage1.price(tau2)

                # price of firm 2 after scheduling
                df.loc[sim, "price_schedule2"] = self.stage1.price_schedule(tau2)
                
                #  price of firm 1 before firm 2 announcement
                df.loc[sim, "price_pre"] = df.loc[sim, "price2"]

                # price of firm 1 following firm 2 announcement
                b = self.stage1.bdy(tau2)
                df.loc[sim, "price_post"] = self.stage2.price(tau2, b, x2, tau2)

                # price of firm 1 before its announcement
                tau1 = max(self.stage2.time(tau2, b, x2, x1), theta1)
                df.loc[sim, "tau1"] = tau1 
                df.loc[sim, "price1"] = self.stage2.price(tau2, b, x2, tau1)

                # price of firm 1 after scheduling
                df.loc[sim, "price_schedule1"] = self.stage2.price_schedule(tau2, b, x2, tau1)

        
        return df
