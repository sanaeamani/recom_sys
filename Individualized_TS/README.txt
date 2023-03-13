This folder includes test.ipynb notebook which is written to showcase how training and predication work in one round of learning in Cas TS algorithm. In the training step, the Cas TS recommendation system gets an input file including the recommendations and the selected ones by some users and saves new trained parameters. In the prediction step, a set of new recommendations for new users given the trained parameters is returned. 



This folder includes test_simulations.ipynb notebook which is written to run simulations for Cas TS algorithm for multiple rounds/days of learning and to plot the regret and repetition curves.

The class of cas_TS in this folder uses user-ids to account for individualized diversification.

Folder “users_data” is the location where new users data will be saved in simulations.

By downloading the contents in this folder, you’ll be able to test the training and prediction steps of Cas TS algorithm by running test.ipynb. Also, you’ll be able to run simulations for Cas TS to see its regret and repetition curves by running test_simulations.ipynb. 
