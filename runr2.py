import SAGMill.SAGMillAnalysis
import pandas as pd
pd.set_option('display.width',1000)
sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()
results = SAGMill.SAGMillAnalysis.doallpredict(sag, mode='valid')
trainresults = SAGMill.SAGMillAnalysis.doallpredict(sag, mode='train')

valid_r2, valid_r2adj = SAGMill.SAGMillAnalysis.rsquared(sag, results, mode='valid')
train_r2, train_r2adj = SAGMill.SAGMillAnalysis.rsquared(sag, trainresults, mode='train')

print 'validation data adjusted R2:'
print valid_r2adj
print 'training data adjusted R2:'
print train_r2adj
