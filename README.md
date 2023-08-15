The current repository contains source code that showcases the GRRNN model for Causal Time-series Forecasting.
The starting point for optimizing the hyperparameters, training and testing the model is the script 'run.py'.

Command-line options:
 --dataset [ETD|SML2010|AirQuality|energyco|poll]
 --mode [optimize|train|test]
 --horizon: [N]

Example:

- To optimize the hyperparameters for forecasting 7 steps ahead for dataset 'SML2010' execute:
  python run.py --dataset SML2010 --horizon 7 --mode optimize
  The optimal hyperparameters are output to the file 'optimized_configuration.txt'.
  Note: 'configuration_space.txt' contains the range and initial value for each hyperparameter. 
  The range and initial value can be altered with the latter falling in the former.

- To train the model with the optimized hyperparameters produced in the previous step execute:
  python run.py --dataset SML2010 --horizon 7 --mode train
  The train model is output to the file 'model.pt'

- To test the trained model produced in the previous step execute:
  python run.py --dataset SML2010 --horizon 7 --mode test

