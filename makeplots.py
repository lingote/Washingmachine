import SAGMill.SAGMillAnalysis
import pandas as pd
pd.set_option('display.width',1000)
sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()

SAGMill.SAGMillAnalysis.normalize(sag)
SAGMill.SAGMillAnalysis.savehists(sag)
