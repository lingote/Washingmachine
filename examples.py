import SAGMill.SAGMillAnalysis
import pandas as pd
pd.set_option('display.width',1000)

sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()
# run a grid eval on a neural network
pipeline = sag.nngrideval(offset=3, savemodel=True, logfile='test_basic.log')
pllist = ['skpipeline_modelPowerDrawMW_3min.pkl']
# Make prediction. This will return a DataFrame for target variable, one column for each minute
res = sag.singlevarnnpred(pipelinelist=pllist, target='PowerDrawMW')
# Calculate resiudals and save plots
sag.drawsingleresiduals(res, offset=3, mode='valid', target='PowerDrawMW', modeldir='nnet')
