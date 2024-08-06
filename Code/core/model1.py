__metaclass__ = type

from scipy.stats import norm, multivariate_normal
import pandas as pd 
import numpy as np
from core.core import *

class Stage1(Parameters):
    def __init__(self, mu, mustar, sigma, rho, lam, rsquared):
        super(Stage1, self).__init__(mu, mustar, sigma, rho, lam, rsquared)

    def probs(self, t):
        z = (self.price1(t) - self.mu) / self.sigma
        q1 = (1 - t) / (1 - t + t*Phi(z))
        return q1, 1-q1

    def rn_probs(self, t):
        z = (self.price1(t) - self.mustar) / self.sigma
        q1 = (1 - t) / (1 - t + t*Phi(z))
        return q1, 1-q1
 
    def price1(self, t):
        return self.mustar + self.sigma*finv(t) 

    def price2(self, t):
        return (1-self.rho)*self.mustar + self.rho*self.price1(t)

    def price_schedule(self, t):
        q1, q2 = self.rn_probs(t)
        p = self.price1(t)
        z = (p - self.mustar) / self.sigma
        zdot = 1 / fprime(z)
        p1 = - phi(z) * zdot / Phi(z)
        p2 = Phi(-z) / (1-t)
        e1 = p
        e2 = self.mustar + self.sigma * phi(z) / Phi(-z)
        return (p1 * e1 * q1 + p2 * e2 * q2) / (p1 * q1 + p2 * q2)

    def time(self, x):
        z = (x - self.mustar) / self.sigma
        return f(z) 

    def density(self, t, x):
        b = (self.price1(t) - self.mu) / self.sigma
        q1, q2 = self.probs(t)
        z = (x - self.mu) / self.sigma
        out = q1 * phi(z) / self.sigma 
        if z < b:
            out += q2 * phi(z) / (self.sigma * Phi(b)) 
        return out

    def rn_density(self, t, x):
        b = (self.price1(t) - self.mustar) / self.sigma
        q1, q2 = self.rn_probs(t)
        z = (x - self.mustar) / self.sigma
        out = q1 * phi(z) / self.sigma 
        if z < b:
            out += q2 * phi(z) / (self.sigma * Phi(b)) 
        return out
 
    def mean1(self, t):
        q1, q2 = self.probs(t)
        b = (self.price1(t) - self.mu) / self.sigma
        return q1 * self.mu + q2 * (self.mu - self.sigma*truncated_mean(b)) 

    def mean2(self, t):
        return (1-self.rho)*self.mu + self.rho*self.mean1(t)

    def price_of_w(self, t):
        return self.delta - self.kappa*self.sigz**2 + self.gamma * (self.price1(t) + self.price2(t))

    def mean_of_w(self, t):
        return self.delta + self.gamma * (self.mean1(t) + self.mean2(t))

    # x1**2 = mu**2 - 2*mu*sigma*e1 + sigma**2 * e1**2
    def mean_of_square1(self, t):
        q1, q2 = self.probs(t)
        b = (self.price1(t) - self.mu) / self.sigma
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
    def theta(self, numsims):
        return np.random.uniform(low=0, high=1 - 1.0e-6, size=(numsims, 1))
    def x(self, numsims):
        corr = np.array([[1, self.rho], [self.rho, 1]])
        cov = self.sigma**2 * corr
        return multivariate_normal.rvs([self.mu, self.mu], cov, size=numsims)
    def sim(self, numdays, numsims):

        theta = self.theta(numsims)
        x = self.x(numsims)
        grid = np.linspace(0, 0.9999, numdays + 1)
        grid = np.round(grid, 3)
        price1 = pd.Series([self.stage1.price1(t) for t in grid], index=grid)
        price2 = pd.Series([self.stage1.price2(t) for t in grid], index=grid)
        sim_price1 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        sim_price2 = pd.DataFrame(np.nan, dtype=float, index=grid, columns=range(numsims))
        for sim in range(numsims):
            theta1 = theta[sim]
            x1 = x[sim, 0]
            x2 = x[sim, 1]
            tau1 = self.stage1.time(x1)
            tau = max(theta1, tau1)

            # pre first announcement

            # last date in grid before first announcement
            tau_in_grid = np.max([t for t in grid if t < tau])
            sim_price1.loc[:tau_in_grid, sim] = price1.loc[:tau_in_grid]
            sim_price2.loc[:tau_in_grid, sim] = price2.loc[:tau_in_grid]

            # first date in grid after first announcement
            tau_in_grid = np.min([t for t in grid if t >= tau]) 

            # firm 1's price is x1 after disclosing
            sim_price1.loc[tau_in_grid:, sim] = x1
            sim_price2.loc[tau_in_grid:, sim] = self.rho*x1 + (1-self.rho)*self.mustar

            if (sim+1) % 1000 == 0:
                print(f"sim {sim+1}") 

        sim_mkt = (
            self.delta 
            - self.kappa * self.sigz**2
            + self.gamma * (sim_price1 + sim_price2)
        )
        theta = pd.DataFrame(theta, columns=["theta1"])
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
 
    def theta(self, numsims):
        return np.random.uniform(low=0, high=1 - 1.0e-6, size=(numsims, 1))

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
        df.theta1 = theta
        df.x1 = x[:, 0]
        df.x2 = x[:, 1]
        df.theta2 = 1
        df.tau2 = 1
       
        for sim in range(numsims):

            if (sim+1) % 1000 == 0: print(sim+1)

            theta1 = theta[sim]
            x1 = x[sim, 0]
            x2 = x[sim, 1]
            tau1 = max(self.stage1.time(x1), theta1)
                    
            # price of firm 1 before announcement
            df.loc[sim, "tau1"] = tau1
            df.loc[sim, "price1"] = self.stage1.price1(tau1)

            # price of firm 1 after scheduling
            df.loc[sim,"price_schedule1"] = self.stage1.price_schedule(tau1)

            # price of firm 2 before firm 1 announcement
            df.loc[sim, "price_pre"] = self.rho*df.loc[sim, "price1"] + (1-self.rho)*self.mustar
            
            # price of firm 2 after firm 1 announcement
            df.loc[sim, "price_post"] = self.rho*x1 + (1-self.rho)*self.mustar

            # price of firm 2 before its announcement
            df.loc[sim, "price2"] = df.loc[sim, "price_post"]

            # price of firm 2 after scheduling
            df.loc[sim, "price_schedule2"] = df.loc[sim, "price_post"]
         
        return df