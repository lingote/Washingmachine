#-*- encoding: UTF-8 -*-
"""
Prototype module for SAG mill analysis
"""
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.formula.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


seed = 1234
np.random.seed(seed)

class SAGMillAnalyzer():
    """
    Main class for our analysis
    """
    def __init__(self, indata='/run/media/ignacio/data/intellisense/sag/SAG_data.csv'):
        self._indata = indata
        #self.df['Time'] = self.df['Time'].apply(lambda x: pd.to_datetime(x))
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
        #self.df = pd.read_csv(indata, usecols=range(1,15))
        self.df = pd.read_csv(indata, parse_dates=['Time'], index_col='Time', date_parser=dateparse)
        self.df.drop('Unnamed: 0', axis=1, inplace=True)
        self.df = self.df.dropna()
        # rename columns, use shorter names, w/o spaces
        self.orignames = self.df.columns
        newcols = ['PressA', 'PressB', 'PressC', 'PressD', 'ConvBeltPSD', 'ConvBeltFines',
                   'ConvFeedRate', 'DilutionFlow', 'Torque', 'PowerDrawMW', 'SCATSConvBelt',
                   'Speed', 'Anomaly']
        self.df.columns = newcols
        self.dfdata = {}
        self.dfdata['test'] = self.df['2016-04-01':self.df.index[-1]]
        self.dfdata['train'] = self.df.drop(self.dfdata['test'].index)
        # take 2/3 for training, 1/3 for validation
        ntrain = self.dfdata['train'].shape[0]/3
        self.dfdata['valid'] = self.dfdata['train'].loc[self.dfdata['train'].index[2*ntrain:]]
        self.dfdata['train'] = self.dfdata['train'].loc[self.dfdata['train'].index[:2*ntrain]]
        # define a scaler object
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        #self.scaler = StandardScaler()
        self.scaler = self.scaler.fit(self.dfdata['train'])
        self.controlvars = ['Speed (RPM)', 'Conveyor Belt Feed Rate (t/h)',
                            'Dilution Flow Rate (m3/h)']
        self.feedvars = ['Conveyor Belt PSD +4 (%)', 'Conveyor Belt PSD Fines (%)']
        #self.perfvars = ['Power Draw (MW)', 'Motor Torque (%)', 'Bearing Pressure A (kPa)',
        #            'Bearing Pressure B (kPa)', 'Bearing Pressure C (kPa)',
        #            'Bearing Pressure D (kPa)', 'SCATS Conveyor Belt Feed Rate (t/h)']
        self.perfvars = ['PressA', 'PressB', 'PressC', 'PressD',
                         'Torque', 'PowerDrawMW', 'SCATSConvBelt']


def gettraindata(sag, mode='train', targetvar='PowerDrawMW', offset=10):
    """
    Build dataset with target value <offset> minutes ahead
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    yfut = (pd.Series(sag.dfdata[mode].loc[sag.dfdata[mode].index[offset:],
                                           targetvar].values,
                      index=sag.dfdata[mode]
                      .index[:-offset]))
    yfut.name = '{}Pred'.format(targetvar)
    return (pd.concat([sag.dfdata[mode].loc[:sag.dfdata[mode].index[-offset-1],
                                            :], yfut], copy=False, axis=1))


def createforecast(sag, targetvar='PowerDrawMW', stepsahead=10):
    """
    Create forecast for targetvar with 'stepsahead' of X
    """
    lr = linear_model.LinearRegression()
    targetname = '{}Pred'.format(targetvar)
    traindata = gettraindata(sag, mode='train', targetvar=targetvar, offset=stepsahead)
    #linregfit = lr.fit(sag.train.loc[:sag.train.index[-stepsahead-1],:], yfut)
    linregfit = lr.fit(traindata.drop(targetname, axis=1), traindata[targetname])
    return linregfit


def dotenminuteforecast(sag, target='PowerDrawMW', mode='valid'):
    """
    Create 1-10min forecast for target variable
    :param sag: input SAG object
    :param target: target var
    :param mode: either 'valid', 'train' or 'test'
    :return: array with LinearRegression objects (fits) and predicted values
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    fits = []
    for i in xrange(1, 11):
        fits.append(createforecast(sag, targetvar=target, stepsahead=i))
        fits[-1].minute = i
    valpred = (pd.concat([pd.DataFrame(i.predict(sag.dfdata[mode]),
                                       index=sag.dfdata[mode].index)
                          for i in fits], axis=1, copy=False))
    valpred.columns = ['{}min'.format(i) for i in range(1, 11)]
    return fits, valpred


def doallpredict(sag, mode='valid'):
    """
    Run all minute-wise predictions on all performance variables
    :param sag: input SAG object
    :return: dictionary with forecast values for all performance variables
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    results = {}
    for i in sag.perfvars:
        results[i] = dotenminuteforecast(sag, i, mode)
    return results


def rsquared(sag, results, mode='train'):
    """
    Get R^2 for all performance variable prediction for every
    1-10 minute forecast.
    :param sag: input SAG object
    :param results: dict with perf variable as key, value is tuple of
                    fit results and prediction values
    :return: dictionary with forecast values for all performance variables
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    r2df = pd.DataFrame(index=sag.perfvars)
    r2adjusteddf = pd.DataFrame(index=sag.perfvars)
    for i in sag.perfvars:
        r2 = []
        for j in results[i][1]:
            offset = int(j.strip('min'))
            yhat = results[i][1][j]
            yhat = yhat.loc[yhat.index[:-offset]]
            y = sag.dfdata[mode].loc[sag.dfdata[mode].index[offset:], i]
            sse = sum((y.values - yhat.values)**2)
            sstotal = sum((y - np.mean(y))**2)
            r_squared = 1 - (float(sse))/sstotal
            nregressor = sag.dfdata[mode].shape[1]
            adjusted_r_squared = (r_squared - (1 - r_squared)*nregressor
                                  /(len(y.values) - nregressor - 1))
            r2df.loc[i, j] = r_squared
            r2adjusteddf.loc[i, j] = adjusted_r_squared
    return r2df, r2adjusteddf


def calcprederrormin(sag, pred, mode='valid', offset=1, target='PowerDrawMW'):
    """
    Compute difference between validation
    data and prediction at <offset> minutes.
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    dfdata = sag.dfdata[mode]
    pidx1 = pred.index[0]
    pidx2 = pred.index[-(1+offset)]
    vidx1 = dfdata.index[offset]
    vidx2 = dfdata.index[-1]
    # Create col name
    offsetcol = '{}min'.format(offset)
    diffpred = (sag.dfdata[mode].loc[vidx1:vidx2, target].values
                - pred.loc[pidx1:pidx2, offsetcol].values)
    return diffpred


def calcresiduals(sag, pred, mode='valid', offset=1, target='PowerDrawMW'):
    """
    Compute difference between validation
    data and prediction at <offset> minutes.
    :return: Dataframe with observered y, and yerr
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    dfdata = sag.dfdata[mode]
    pidx1 = pred.index[0]
    pidx2 = pred.index[-(1+offset)]
    vidx1 = dfdata.index[offset]
    vidx2 = dfdata.index[-1]
    # Create col name
    print vidx1, vidx2
    offsetcol = '{}min'.format(offset)
    print offsetcol
    diffpred = pd.DataFrame(index=pd.date_range(vidx1, vidx2, freq='min'))
    diffpred = pd.DataFrame(index=sag.dfdata[mode].index[offset:])
    diffpred['yobserved'] = sag.dfdata[mode].loc[vidx1:vidx2, target].values
    diffpred['yerr'] = (sag.dfdata[mode].loc[vidx1:vidx2, target].values
                             - pred.loc[pidx1:pidx2, offsetcol].values)
    return diffpred


def drawresiduals(sag, pred, mode='valid'):
    """
    Draw residuals vs observed value
    """
    for i in sag.perfvars:
        for j in range(1,11):
            residuals = calcresiduals(sag, pred[i][1], mode=mode, offset=j, target=i)
            bin_mean, bin_edges, bin_num = binned_statistic(residuals.ix[:,0], residuals.ix[:,1], statistic='mean', bins=50)
            bin_std, bin_stdedges, bin_stdnum = binned_statistic(residuals.ix[:,0], residuals.ix[:,1], statistic='std', bins=50)
            plt.scatter(residuals.ix[:,0], residuals.ix[:,1],zorder=1, s=1, color='gray')
            plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
            plt.hlines(bin_mean+bin_std, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
            plt.hlines(bin_mean-bin_std, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
            bin_mid = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.
            plt.vlines(bin_mid, bin_mean-bin_std, bin_mean+bin_std, colors='b', lw=1,label='binned statistic of data',zorder=2)
            plt.title('{} {}min residuals'.format(i, j))
            plt.xlabel('{}'.format(i))
            plt.ylabel('Residuals')
            plt.savefig('residual{}{}.png'.format(i,j))
            plt.close()



def calcallerrors(sag, preddict, mode='valid'):
    """
    Calculate errors for 1...10 min predictions
    wrt validation sample
    :param sag: sag object
    :param preddict: prediction dict. Key per performance variable
    :param target: target variable
    :param mode: error on train, valid or test
    :return: dictionary with prediction errors with perf vars and minute as keys
    """
    prederr = {}
    for i in sag.perfvars:
        errdf = {}
        for j in range(1, 11):
            errdf['{}min'.format(j)] = (calcprederrormin(sag, preddict[i][1],
                                                         mode=mode, offset=j,
                                                         target=i))
        prederr[i] = errdf
    return prederr


###def calcresiduals(sag, preddict, mode='valid'):
###    """
###    Calculate errors for 1...10 min predictions
###    wrt validation sample
###    :param sag: sag object
###    :param preddict: prediction dict. Key per performance variable
###    :param target: target variable
###    :param mode: error on train, valid or test
###    :return: dictionary with prediction errors with perf vars and minute as keys
###    """
###    prederr = {}
###    for i in sag.perfvars:
###        errdf = {}
###        for j in range(1, 11):
###            errdf['{}min'.format(j)] = (calcprederrormin(sag, preddict[i][1],
###                                                         mode=mode, offset=j,
###                                                         target=i))
###        prederr[i] = errdf
###    return prederr


def plotprederr(preddict):
    """
    Make histogram of prediction errors
    :param preddict: dict with perf vars = key, val = pred error
    :return: saves histograms as pngs
    """
    for k in preddict.iterkeys():
        for k2, v2 in preddict[k][1].iteritems():
            plt.hist(v2, 100)
            plt.title('{} {} pred error'.format(k, k2))
            plt.yscale('log', nonposy='clip')
            plt.savefig('{}_{}_pred_error.png'.format(k, k2))
            plt.close()

def plotpredvsdatamin(sag, pred, offset=1, target='PowerDrawMW'):
    """
    Plot prediction vs validation data
    """
    if offset > 10:
        print "Offset must be <= 10"
        return
    pidx1 = pred.index[0]
    pidx2 = pred.index[-(1+offset)]
    vidx1 = sag.valid.index[offset]
    vidx2 = sag.valid.index[-1]
    # Create col name
    offsetcol = '{}min'.format(offset)
    fig, ax = plt.subplots()
    (ax.plot(sag.valid.loc[vidx1:vidx2, target].values,
             pred.loc[pidx1:pidx2, offsetcol].values, 'o'))
    plt.title("{}: Predicted vs data (validation)".format(target))
    plt.xlabel("Data {}".format(target))
    plt.ylabel("Predicted {}".format(target))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    (ax.set_xlim([sag.valid.loc[vidx1:vidx2, target].min(),
                  sag.valid.loc[vidx1:vidx2, target].max()]))
    plt.show()


def normalize(sag, inverse=False):
    """
    Transform dataset using the scaler object attached to the SAG object
    """
    colnames = sag.dfdata['train'].columns
    trainindex = sag.dfdata['train'].index
    validindex = sag.dfdata['valid'].index
    testindex = sag.dfdata['test'].index
    for i in sag.dfdata.iterkeys():
        if inverse:
            sag.dfdata[i] = (pd.DataFrame(sag.scaler
                                          .inverse_transform(sag.dfdata[i])
                                          , columns=colnames))
        else:
            sag.dfdata[i] = (pd.DataFrame(sag.scaler.transform(sag.dfdata[i])
                                          , columns=colnames))
    sag.dfdata['train'].index = trainindex
    sag.dfdata['test'].index = testindex
    sag.dfdata['valid'].index = validindex


def dohistograms(df, dfcol):
    """
    Create simple histograms for exploratory analysis
    """
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins, patches = plt.hist(df[dfcol], 100, normed=1, facecolor='blue', alpha=0.75)
    dfmin = df[dfcol].dropna().min()
    dfmax = df[dfcol].dropna().max()
    dfmean = df[dfcol].dropna().mean()
    dfstd = df[dfcol].dropna().std()
    print 'working on histogram {}'.format(dfcol)
    plt.xlabel(dfcol)
    plt.ylabel('a.u.')
    plt.title('{}'.format(dfcol))
    plt.yscale('log', nonposy='clip')
    #plt.text(dfmin + (dfmax-dfmin)/2., 1, 'mean {}'.format(dfmean), fontsize=15,
    plt.text(0.2, 0.85, 'mean={}'.format(dfmean), fontsize=15,
             transform=ax.transAxes)
    plt.text(0.2, 0.8, 'std={}'.format(dfstd), fontsize=15,
             transform=ax.transAxes)
    plt.savefig('hist{}.png'.format(dfcol.upper()))


def dotimeseries(df, dfcol):
    """
    Plot time series of variables
    """
    plt.close('all')
    ts = df[dfcol]
    plt.xlabel('time')
    plt.ylabel(dfcol)
    plt.title('TimeSeries of {}'.format(dfcol))
    plt.plot(ts)
    plt.savefig('timeseries{}.png'.format(dfcol.upper()))


def savehists(sag):
    """
    Create and save histograms and graphs of training data
    :return:
    """
    for i in sag.dfdata['train'].columns.values:
        dohistograms(sag.dfdata['train'], i)

    for i in sag.dfdata['train'].columns.values:
        #plt.close()
        dotimeseries(sag.dfdata['train'], i)
        #plt.close('all')


def doautocorrplot(sag, predictor, outname='ACFPlots.pdf'):
    """
    plot autocorrelation
    :param df: the dataframe
    :param predictor: the predictor variable
    :return:
    """
    with PdfPages(outname) as pp:
        for i in sag.perfvars + sag.controlvars + sag.feedvars:
            plot_acf(sag.dfdata['train'][i], lags=10000)
            plt.title('ACF {}'.format(i))
            pp.savefig()
            plt.close('all')


def test_stationarity(timeseries):

    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)
    roller = timeseries.rolling(window=12, center=False)
    rolmean = roller.mean()
    rolstd = roller.std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = (pd.Series(dftest[0:4],
                          index=['Test Statistic', 'p-value',
                                 '#Lags Used', 'Number of Observations Used']))
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


def baseline_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    # create model
    model = Sequential()
    #model.add(Dropout(dropout, input_shape=(13,)))
    model.add(Dense(13, input_dim=13, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def evalnn(sag):
    # evaluate model with standardized dataset
    tvar = 'PowerDrawMW'
    traindata = gettraindata(sag, mode='train', targetvar=tvar, offset=5)
    tvar = '{}Pred'.format(tvar)
    seed = 1234
    np.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    #estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=10000, verbose=0)))
    #estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, batch_size=100000, verbose=0)))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=150, optimizer='rmsprop', init='normal', batch_size=10000, verbose=0)))
    print estimators[-1][1].get_params().keys()
    X = traindata.drop(tvar, axis=1)
    Y = traindata[tvar]
    pipeline = Pipeline(estimators)
    dogridcv = False
    if dogridcv:
        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = [50, 100, 150]
        epochs = [50, 75, 125]
        #batches = [5, 10, 20]
        #param_grid = dict(mlp__optimizer=optimizers, mlp__epochs=epochs, mlp__batch_size=batches, mlp__init=init)
        param_grid = dict(mlp__optimizer=optimizers, mlp__epochs=epochs, mlp__init=init)
        print pipeline.get_params().keys()
        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid)
        grid_result = grid.fit(X, Y)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    else:
        kfold = KFold(n_splits=5, random_state=seed)
        results = cross_val_score(pipeline, traindata.drop(tvar, axis=1), traindata[tvar], cv=kfold)
        print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

if __name__ == '__main__':
    sag = SAGMillAnalyzer()
    sag.savehists()
