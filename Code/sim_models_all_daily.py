#############################################################################
#
# This file simulates daily returns for creating the Kapadia-Zekhnini figure.
#
# Created by Kerry Back, July 2024
#
# ###########################################################################

from core.model1 import DailySim as Sim1
from core.model2 import DailySim as Sim2
from core.model3 import DailySim as Sim3

DIR = "./simulations"

mu = 105
mustar = 100
sigma = 15
lam = 1000
rsquared = 0.2
rho = 0.5

sim1 = Sim1(mu, mustar, sigma, rho, lam, rsquared)
sim2 = Sim2(mu, mustar, sigma, rho, lam, rsquared)
sim3 = Sim3(mu, mustar, sigma, rho, lam, rsquared)

numdays = 150
numsims = 100000

print("Model 1")
theta, x, sim_price1, sim_price2, sim_mkt = sim1.sim(numdays=numdays, numsims=numsims)
theta.to_csv(f"{DIR}/model1_daily_theta.csv")
x.to_csv(f"{DIR}/model1_daily_x.csv")
sim_price1.to_csv(f"{DIR}/model1_daily_price1.csv")
sim_price2.to_csv(f"{DIR}/model1_daily_price2.csv")
sim_mkt.to_csv(f"{DIR}/model1_daily_mkt.csv")

print("\nModel 2")
theta, x, sim_price1, sim_price2, sim_mkt = sim2.sim(numdays=numdays, numsims=numsims)
theta.to_csv(f"{DIR}/model2_daily_theta.csv")
x.to_csv(f"{DIR}/model2_daily_x.csv")
sim_price1.to_csv(f"{DIR}/model2_daily_price1.csv")
sim_price2.to_csv(f"{DIR}/model2_daily_price2.csv")
sim_mkt.to_csv(f"{DIR}/model2_daily_mkt.csv")

print("\nModel 3")
theta, x, sim_price1, sim_price2, sim_mkt = sim3.sim(numdays=numdays, numsims=numsims)
theta.to_csv(f"{DIR}/model3_daily_theta.csv")
x.to_csv(f"{DIR}/model3_daily_x.csv")
sim_price1.to_csv(f"{DIR}/model3_daily_price1.csv")
sim_price2.to_csv(f"{DIR}/model3_daily_price2.csv")
sim_mkt.to_csv(f"{DIR}/model3_daily_mkt.csv")