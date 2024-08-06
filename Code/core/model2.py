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

    def probs(self, t, b, x2, u):
        mu1 = self.rho * x2 + (1-self.rho) * self.mu
        sig1 = self.sigma*np.sqrt(1-self.rho**2)
        p = self.price(t, b, x2, u)
        q1 = t * Phi((b-mu1)/sig1)
        q2 = (u-t)*Phi((p-mu1)/sig1)
        q3 = 1 - u
        sum = q1 + q2 + q3 
        return q1 / sum, q2 / sum, q3 / sum

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

    # stage 2 for firm 2 if firm 1 discloses first

    def price2(self, x1):
        return self.rho*x1 + (1-self.rho)*self.mustar

    def price_schedule2(self, x1):
        return self.price2(x1)

def Solve(t, rho):
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

fixed_points = pd.read_csv('fixed_points/model2.csv', index_col="time")
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

    def probs(self, t):
        b = (self.bdy(t) - self.mu) / self.sigma
        q1 = (1 - t) / (1 - t + t*Phi(b))
        return q1, 1-q1

    def rn_probs(self, t):
        b = (self.bdy(t) - self.mustar) / self.sigma
        q1 = (1 - t) / (1 - t + t*Phi(b))
        return q1, 1-q1

    def price1(self, t):
        q1, q2 = self.rn_probs(t)
        b = (self.bdy(t) - self.mustar) / self.sigma
        return q1 * self.mustar + q2 * (self.mustar - self.sigma*truncated_mean(b))
       
    def price2(self, t):
        return (1-self.rho)*self.mustar + self.rho*self.price1(t)

    def price_schedule1(self, t):
        q1, q2 = self.rn_probs(t)
        b = self.bdy(t)
        z = (b - self.mustar) / self.sigma
        zdot = derivative(lambda u: self.bdy(u) / self.sigma, t)
        p1 = - phi(z) * zdot / Phi(z)
        p2 = Phi(-z) / (1-t)
        e1 = b
        e2 = self.mustar + self.sigma * phi(z) / Phi(-z)
        return (p1 * e1 * q1 + p2 * e2 * q2) / (p1 * q1 + p2 * q2)

    def price_schedule2(self, t):
        return self.price2(t)

    def mean1(self, t):
        q1, q2 = self.probs(t)
        b = (self.bdy(t) - self.mu) / self.sigma
        return q1 * self.mu + q2 * (self.mu - self.sigma*truncated_mean(b)) 

    def mean2(self, t):
        return (1-self.rho)*self.mu + self.rho*self.mean1(t)

    def price_of_w(self, t):
        return self.delta - self.kappa*self.sigz**2 + self.gamma * (self.price1(t) + self.price2(t))

    def mean_of_w(self, t):
        return self.delta + self.gamma * (self.mean1(t) + self.mean2(t))

    def mean_of_square1(self, t):
        p = (self.price1(t) - self.mu) / self.sigma
        num = t*(2*self.sigma*self.mu + self.sigma**2 * p) * phi(p)
        den = 1 - t + t*Phi(p)
        return  self.mu**2 + self.sigma**2 - num / den

    # x1**2 = mu**2 - 2*mu*sigma*e1 + sigma**2 * e1**2
    def mean_of_square1(self, t):
        q1, q2 = self.probs(t)
        b = (self.bdy(t) - self.mu) / self.sigma
        return (
            q1 * (self.mu**2 + self.sigma**2)
            + q2 * (self.mu**2 - 2*self.mu*self.sigma*truncated_mean(b) + self.sigma**2*truncated_square(b))
        )

    def var1(self, t):
        return self.mean_of_square1(t) - self.mean1(t)**2

    # w = delta + gamma * (x1 + (1-rho)*mu + rho*x1 + e1) + z
    def varw(self, t):
        return (
            self.gamma**2 * (1+self.rho)**2 * self.var1(t)
            + self.gamma**2 * (1-self.rho**2) * self.sigma**2 
            + self.sigz**2
        )

    def cov1w(self, t):
        return self.gamma*(1+self.rho)*self.var1(t)

    # x2 = (1-rho)*mu + rho*x1 + e
    # w = delta + gamma*(1+rho)*x1 + gamma*e + z
    def cov2w(self, t):
        return self.gamma*self.rho*(1+self.rho)*self.var1(t) + self.gamma*(1-self.rho**2)*self.sigma**2

    def beta1(self, t):
        return self.cov1w(t) / self.varw(t)

    def beta2(self, t):
        return self.cov2w(t) / self.varw(t)
    
    def alpha1(self, t):
        return self.mean1(t) - self.price1(t) - self.beta1(t) * (self.mean_of_w(t) - self.price_of_w(t))

    def alpha2(self, t):
        return self.mean2(t) - self.price2(t) - self.beta2(t) * (self.mean_of_w(t) - self.price_of_w(t))

    def alpha1_wrong(self, t):
        return self.mean1(t) - self.price1(t) - self.beta1(0) * (self.mean_of_w(t) - self.price_of_w(t))

    def alpha2_wrong(self, t):
        return self.mean2(t) - self.price2(t) - self.beta2(0) * (self.mean_of_w(t) - self.price_of_w(t))

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
        price1 = pd.Series(self.stage1.price1(grid), index=grid)
        price2 = pd.Series(self.stage1.price2(grid), index=grid)
        sim_price1 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        sim_price2 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        for sim in range(numsims):
            theta1 = theta[sim, 0]
            theta2 = theta[sim, 1]
            x1 = x[sim, 0]
            x2 = x[sim, 1]
            tau1 = self.stage1.time(x1)
            tau = min(theta2, max(theta1, tau1))

            # pre first announcement

            # last date in grid before first announcement
            tau_in_grid = np.max([t for t in grid if t < tau])
            sim_price1.loc[:tau_in_grid, sim] = price1.loc[:tau_in_grid]
            sim_price2.loc[:tau_in_grid, sim] = price2.loc[:tau_in_grid]

            # first date in grid after first announcement
            tau_in_grid = np.min([t for t in grid if t >= tau]) 

            # firm 1 discloses first
            if max(theta1, tau1) < theta2:
        
                # firm 1's price is x1
                sim_price1.loc[tau_in_grid:, sim] = x1

                # first date in grid after theta2
                theta2_in_grid = np.min([t for t in grid if t >= theta2])
        
                # firm 2's price is weighted average and then x2
                if theta2_in_grid > tau_in_grid:
                    sim_price2.loc[tau_in_grid: theta2_in_grid, sim] = self.rho * x1 + (1 - self.rho) * self.mustar
                sim_price2.loc[theta2_in_grid:, sim] = x2
       
            # firm 2 discloses first
            else:
                # boundary at firm 2's disclosure time 
                B = self.stage1.bdy(theta2)

                # firm 2's price is x2
                sim_price2.loc[tau_in_grid:, sim] = x2

                # first date in grid after theta1
                theta1_in_grid = min([t for t in grid if t >= theta1])

                # first date in grid after firm 1 would voluntarily disclose
                d = np.min(
                    [
                        u for u in grid 
                        if u >= tau_in_grid 
                        and u >= theta1_in_grid
                        and x1 >= self.stage2.price(theta2, B, x2, u)
                    ]
                )

                # firm 1's price is from stage2 and then x1
                period = [u for u in grid if u>=tau_in_grid and u<d]
                for u in period:
                    sim_price1.loc[u, sim] = self.stage2.price(theta2, B, x2, u)
                sim_price1.loc[d:, sim] = x1
     
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
                "price_schedule1", "price_schedule2"]
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
                    
            # firm 1 discloses first
            if tau1 < theta2:

                # price of firm 1 before announcement
                df.loc[sim, "tau1"] = tau1
                df.loc[sim, "price1"] = self.stage1.price1(tau1)

                # price of firm 1 after scheduling
                df.loc[sim, "price_schedule1"] = self.stage1.price_schedule1(tau1)
                
                # price of firm 2 before firm 1 announcement
                df.loc[sim, "price_pre"] = self.rho*df.loc[sim, "price1"] + (1-self.rho)*self.mustar
                
                # price of firm 2 following firm 1 announcement
                df.loc[sim, "price_post"] = self.rho*x1 + (1-self.rho)*self.mustar

                # price of firm 2 before its announcement
                df.loc[sim, "tau2"] = theta2
                df.loc[sim, "price2"] = self.stage2.price2(x1)

                # price of firm 2 after scheduling
                df.loc[sim, "price_schedule2"] = self.stage2.price_schedule2(x1)
            
            # firm 2 discloses first
            else:

                # price of firm 2 before announcement
                df.loc[sim, "tau2"] = theta2
                df.loc[sim, "price2"] = self.stage1.price2(theta2)

                # price of firm 2 after scheduling
                df.loc[sim, "price_schedule2"] = self.stage1.price_schedule2(theta2)

                # price of firm 1 before firm 2 announcement
                df.loc[sim, "price_pre"] = self.stage1.price1(theta2)

                # price of firm 1 following firm 2 announcement
                b = self.stage1.bdy(theta2)
                df.loc[sim, "price_post"] = self.stage2.price(theta2, b, x2, theta2)

                # price of firm 1 before its announcement
                b = self.stage1.bdy(theta2)
                tau1 = max(self.stage2.time(theta2, b, x2, x1), theta1)
                df.loc[sim, "tau1"] = tau1 
                df.loc[sim, "price1"] = self.stage2.price(theta2, b, x2, tau1)

                # price of firm 1 after scheduling
                df.loc[sim, "price_schedule1"] = self.stage2.price_schedule(theta2, b, x2, tau1)
        
        return df