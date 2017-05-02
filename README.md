Washing machine project
=======================

Goal of this project is the minute-wise prediction of SAG mill performance over the next 10 minutes.
Project is based on data collected over one year at one particular SAG mill. 
Prediction model files saved in h5 and pkl files.
Tested with python 2.7

Python package SAGMill
-------------------------

- SAGData.py: Prepare data, split train, valid and test data
- SAGMillAnalysis.py: run analysis
- SAGModels.py: Various keras models


LSTM results
-------------------------

- lstm/: directory with results of LSTM prediction

Installation
------------

- Install and activate virtualenv. mkvirtualenv recommended. More info here: https://virtualenvwrapper.readthedocs.io/en/latest/
- Install requirements: 
	
	pip install -r requirements.txt


Run LSTM fit and prediction
---------------------------

       python runlstm.py -t SCATSConvBelt -m train -s True -f


Draw residuals using csv with yhat 
----------------------------------

       python drawlstmresiduals.py -t Torque -m valid -f lstmForecast_PowerDrawMW_4neurons_100ep_200batch_valid.csv

