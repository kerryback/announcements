#############################################################################
#
# This file simulates model paths.
#
# Created by Kerry Back, July 2024
#
# ###########################################################################

from core.model1 import Sim as Sim1
from core.model2 import Sim as Sim2
from core.model3 import Sim as Sim3

DIR = "./simulations"

mu = 103
mustar = 100
sigma = 20
lam = 1000
rsquared = 0.2
rho = 0.5

sim1 = Sim1(mu, mustar, sigma, rho, lam, rsquared)
sim2 = Sim2(mu, mustar, sigma, rho, lam, rsquared)
sim3 = Sim3(mu, mustar, sigma, rho, lam, rsquared)

numsims = 100000

print("Model 1")
sim1.sim(numsims).to_csv(f"{DIR}/model1_sim.csv")

print("\nModel 2")
sim2.sim(numsims).to_csv(f"{DIR}/model2_sim.csv")

print("\nModel 3")
sim3.sim(numsims).to_csv(f"{DIR}/model3_sim.csv")

print("\nModel 3 with rho=0.1")
rho = 0.1
sim10 = Sim3(mu, mustar, sigma, rho, lam, rsquared)
sim10.sim(numsims).to_csv(f"{DIR}/model3_sim_10.csv")

print("\nModel 3 with rho=0.9")
rho = 0.9
sim90 = Sim3(mu, mustar, sigma, rho, lam, rsquared)
sim90.sim(numsims).to_csv(f"{DIR}/model3_sim_90.csv")