This file contains the code used to solve, run simulations, and generate figures for the paper Discretionary Announcement Timing and Stock Returns by Kerry Back, Bruce Carlin, Seyed Kazempour, and Chloe Xie.

- fixed_points_model2.py and fixed_points_model3.py solve Models II and III respectively on a $(t, \rho)$ grid.  The solutions are saved to csv files.

- sim_models_all.py simulates all three models for parameter values specified in the file and saves the simulations to csv files.

- sim_models_all_daily.py simulates all three models and computes "daily" returns by splitting the time interval $[0, 1]$ into equal sized "days."

- figures_model1.ipynb, figures_model2.ipynb, and figues_model3.ipynb generate figures for each of the models.

- figures_models_all.ipynb generates figures that illustrate multiple models.

- The core module contains definitions of functions and classes that are used for all models (core.py) or for individual models (model1.py, model2.py, and model3.py). 

If needed, the specific versions of packages used when running these files can be found in requirements.txt.