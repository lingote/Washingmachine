import SAGMill.SAGMillAnalysis
import pandas as pd
pd.set_option('display.width',1000)

sag = SAGMill.SAGMillAnalysis.SAGMillAnalyzer()
dfpredict = pd.DataFrame(index=sag.dfdata['valid'].index,
                         columns = ['1min', '2min', '3min', '4min',
                                    '5min', '6min', '7min', '8min',
                                    '9min', '10min'])

perfvars = ['PressA', 'PressB', 'PressC', 'PressD',
            'Torque', 'PowerDrawMW', 'SCATSConvBelt']

target=perfvars[0]

for i in xrange(1,11):
    logfile='{}_{}min_GridSearchCV.log'.format(target, i)
    pipeline = SAGMill.SAGMillAnalysis.nngrideval(sag, offset=i, savemodel=True, target=target, logfile=logfile)
    print 'done with {} minute prediction'.format(i)
