Washing machine project
=======================

Goal of this project is the minute-wise prediction of SAG mill performance over the next 10 minutes.
Project is based on data collected over one year at one particular SAG mill. 
Prediction model files saved in h5 and pkl files.
Tested with python 2.7

Python package SAGMill
-------------------------

- SAGMillAnalysis: main module, load data and split into training, validation and test sets


Installation
------------

- Install and activate virtualenv. mkvirtualenv recommended. More info here: https://virtualenvwrapper.readthedocs.io/en/latest/
- Install requirements: 
	
	pip install -r requirements.txt


Example
-------

 import SAGMill.SAGMillAnalysis
 import pandas as pd
 pd.set_option('display.width',1000)
 sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()
 # compute 3 minute prediction model on target variable 'PowerDrawMW' using a grid search CV on a keras neural network
 pipeline = sag.nngrideval(offset=i, target='PowerDrawMW', logfile='mylog.log')
 # compute 1 to 10 minute-wise predictions on validation data using saved model in pipeline
 yhat = sag.singlevarnnpred(pipelinelist=[pipeline], target='PowerDrawMW', mode='valid')
 # draw residuals and save them in directory 'nnet'
 sag.drawsingleresiduals(yhat, offset=3, mode='valid', target='PowerDrawMW', modeldir='nnet', mode='valid')
