import SAGMill.SAGMillAnalysis
import pandas as pd

sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()
SAGMill.SAGMillAnalysis.normalize(sag)
results = SAGMill.SAGMillAnalysis.doallpredict(sag, mode='valid')
SAGMill.SAGMillAnalysis.plotprederr(results)

