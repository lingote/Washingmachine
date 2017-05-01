#-*- coding: utf-8 -*-
"""
Module for SAG mill analysis. This module loads SAG data and creates models
for prediction of various performance variables. Uses Keras with Theano backend.
"""
import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.formula.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from .SAGUtils import get_median_filtered
from . import SAGModels


SEED = 1234
np.random.seed(SEED)


class ModelInfo(object):
    """
    Simple object holding fit results and scalers
    """
    def __init__(self, modelname, target, offset, neurons,
                 epochs, batch, xscaler, yscaler):
        self.modelname = modelname
        self.target = target
        self.offset = offset
        self.neurons = neurons
        self.epochs = epochs
        self.batch = batch
        self.xscaler = xscaler
        self.yscaler = yscaler


def linregforecast(sag, targetvar='PowerDrawMW', stepsahead=10,
                   savemodel=False, scaler=StandardScaler()):
    """
    Create forecast for targetvar with 'stepsahead' of X
    :param sag: input sag object
    :param targetvar: target variable
    :param stepsahead: minutes ahead
    :return linregfit: fitted LinearRegression object
    """
    lr = linear_model.LinearRegression()
    targetname = '{}Pred'.format(targetvar)
    traindata = sag.gettraindata(mode='train', targetvar=targetvar, offset=stepsahead)
    traindata = traindata.dropna()
    #linregfit = lr.fit(traindata.drop(targetname, axis=1), traindata[targetname])

    estimators = []
    estimators.append(('standardize', scaler))
    sagmodel = 'linReg{}_{}min'.format(targetvar, stepsahead)
    estimators.append((sagmodel, lr))
    X = traindata.drop(targetname, axis=1)
    X = X.drop(targetvar, axis=1)
    Y = traindata[targetname]
    Y = Y.values.reshape(-1, 1)
    yscaler = StandardScaler().fit(Y)
    Yscale = yscaler.transform(Y)
    pipeline = Pipeline(estimators)
    pipeline.fit(X, Yscale)
    # Adding the yscaler to the pipeline
    pipeline.yscaler = yscaler
    if savemodel:
        plfile = 'skpipeline_{}.pkl'.format(sagmodel)
        # add modelfile member to pipeline
        joblib.dump(pipeline, plfile)
        print 'Saved pipeline to {}'.format(plfile)
    return pipeline


def rsquared(sag, results, mode='train'):
    """
    Get R^2 for all performance variable prediction for every
    1-10 minute forecast.
    :param sag: input sag object
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
        for j in results[i]:
            offset = int(j.strip('min'))
            yhat = results[i][j]
            yhat = yhat.loc[yhat.index[:-offset]]
            y = sag[mode].loc[sag[mode].index[offset:], i]
            sse = sum((y.values - yhat.values)**2)
            sstotal = sum((y - np.mean(y))**2)
            r_squared = 1 - (float(sse))/sstotal
            nregressor = sag[mode].shape[1]
            adjusted_r_squared = (r_squared - (1 - r_squared)*nregressor
                                  /(len(y.values) - nregressor - 1))
            r2df.loc[i, j] = r_squared
            r2adjusteddf.loc[i, j] = adjusted_r_squared
    return r2df, r2adjusteddf


def calcresiduals(sag, yhat, mode='valid', offset=1, target='PowerDrawMW', normed=False):
    """
    Compute difference between validation
    data and prediction at <offset> minutes.
    :return: Dataframe with observered y, and yerr
    """
    if mode not in ['test', 'train', 'valid']:
        print 'mode {} is not valid'.format(mode)
        return
    dfdata = sag[mode]
    # Create col name
    offsetcol = '{}min'.format(offset)
    columns = ['yobserved', 'yhat', 'yerr']
    if normed:
        columns.append('yhatnormed')
    diffpred = pd.DataFrame(index=dfdata.index[offset:], columns=columns)
    #diffpred['yobserved'] = dfdata[yhat.index[0]:yhat.index[-offset-1]][target]
    diffpred['yobserved'] = dfdata[offset:][target]
    if normed:
        diffpred['yhatnormed'] = yhat[offsetcol].dropna().values
        diffpred['yhat'] = (yhat.yscalers[offsetcol].
                            inverse_transform(yhat[offsetcol].
                                              dropna().values.reshape(-1, 1)))
    else:
        diffpred['yhat'] = yhat[offsetcol].dropna().values
    diffpred['yerr'] = diffpred['yobserved'] - diffpred['yhat']
    return diffpred


def drawsingleresiduals(sag, yhats, offset=1, mode='valid',
                        target='PowerDrawMW', modeldir='linreg'):
    """
    Draw residuals vs observed value. Save file in subdirectory
    <modeldir>
    :param sag: sag object
    :param yhat: dict with yhat values, keys are perf variables
    :param mode: train, valid, test
    :param modeldir: save graphs in this sub-directory
    """
    residuals = calcresiduals(sag, yhats, mode=mode, offset=offset, target=target)
    residuals.dropna(inplace=True)
    resmean = np.mean(residuals['yerr'])
    resstd = np.std(residuals['yerr'])
    bin_mean, bin_edges, bin_num = binned_statistic(residuals.ix[:, 0],
                                                    residuals.ix[:, 2],
                                                    statistic='mean', bins=50)
    bin_std, bin_stdedges, bin_stdnum = binned_statistic(residuals.ix[:, 0],
                                                         residuals.ix[:, 2],
                                                         statistic='std',
                                                         bins=50)
    plt.scatter(residuals.ix[:, 0], residuals.ix[:, 2], zorder=1, s=1, color='gray')
    plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='b',
               lw=1, label='binned statistic of data', zorder=2)
    plt.hlines(bin_mean+bin_std, bin_edges[:-1], bin_edges[1:],
               colors='b', lw=1, label='binned statistic of data', zorder=2)
    plt.hlines(bin_mean-bin_std, bin_edges[:-1], bin_edges[1:],
               colors='b', lw=1, label='binned statistic of data', zorder=2)
    bin_mid = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.
    plt.vlines(bin_mid, bin_mean-bin_std, bin_mean+bin_std, colors='b',
               lw=1, label='binned statistic of data', zorder=2)
    plt.title('{} {}min residuals using {}'.format(target, offset, model))
    plt.xlabel('{}'.format(target))
    plt.ylabel('yobs - ypred')
    #plt.ylim([-80,70])
    plt.annotate('<r>={:6.3f}'.format(resmean), xy=(0.1, 0.9), xycoords='axes fraction')
    plt.annotate('std(r)={:6.3f}'.format(resstd), xy=(0.1, 0.8), xycoords='axes fraction')
    plt.annotate('{} sample'.format(mode), xy=(0.8, 0.9), xycoords='axes fraction')
    #plt.clf()
    #plt.imshow(heatmap, zorder=1)#, extent = extent)#, origin='lower')
    plt.savefig('{}/residual{}{}_{}.png'.format(modeldir, target, offset, mode))
    plt.close()


def drawresiduals(sag, yhats, target='PowerDrawMW', mode='valid', modeldir='linreg'):
    """
    Draw residuals vs observed value for 1-10min predictions for a single target variable.
    Save file in subdirectory <modeldir>
    :param sag: sag object
    :param yhats: dict with yhat values, keys are perf variables
    :param mode: train, valid, test
    :param modeldir: save graphs in this sub-directory
    """
    for j in range(1, 11):
        sag.drawsingleresiduals(yhats, offset=j, mode=mode, target=target, modeldir=modeldir)


def drawallresiduals(sag, yhats, mode='valid', modeldir='linreg'):
    """
    Draw residuals vs observed value. Save file in subdirectory <modeldir>
    :param sag: sag object
    :param yhat: dict with yhat values, keys are perf variables
    :param mode: train, valid, test
    :param modeldir: save graphs in this sub-directory
    """
    for i in sag.perfvars:
        sag.drawresiduals(yhats[i], target=i, mode=mode, modeldir=modeldir)


def normalize(sag, inverse=False):
    """
    Transform dataset using the scaler object attached to the SAG object
    :param inverse: True if revert normalization
    """
    colnames = sag['train'].columns
    trainindex = sag['train'].index
    validindex = sag['valid'].index
    testindex = sag['test'].index
    for i in sag.dfdata.iterkeys():
        if inverse:
            sag[i] = (pd.DataFrame(sag.scaler
                                   .inverse_transform(sag[i])
                                   , columns=colnames))
        else:
            sag[i] = (pd.DataFrame(sag.scaler.transform(sag[i])
                                   , columns=colnames))
    sag['train'].index = trainindex
    sag['test'].index = testindex
    sag['valid'].index = validindex


def pctchangeplot(sag, target='PowerDrawMW', offset=1):
    """
    Get percent change of variable wrt <offset> minutes in the future
    """
    #pctdata = sag['train'].ix[start:end, target].pct_change(offset).dropna()*100
    pctdata = pd.concat([sag['train'], sag['valid']])[target].pct_change(offset).dropna()*100
    plt.hist(pctdata, 200)
    plt.yscale('log')
    plt.xlabel('rel. change {}min lag (%)'.format(offset))
    plt.title('{} percent change {}min lag'.format(target, offset))
    pltname = 'histPercentChange_{}_{}minutes.png'.format(target, offset)
    plt.savefig(pltname)
    plt.close()


def abschangeplot(sag, target='PowerDrawMW', offset=1):
    """
    Get percent change of variable wrt <offset> minutes in the future
    """
    diffdata = pd.concat([sag['train'], sag['valid']])[target].diff(offset).dropna()
    plt.hist(diffdata, 200)
    plt.yscale('log')
    plt.xlabel('change {}min lag'.format(offset))
    plt.title('{} change {}min lag'.format(target, offset))
    pltname = 'histChange_{}_{}minutes.png'.format(target, offset)
    plt.savefig(pltname)
    plt.close()


def powerdrawdiff(sag, offset=3):
    """
    Plot PowerDraw lag difference for different regimes
    :param offset: lag in minutes
    """
    traindf = sag.gettraindata(mode='train', targetvar='PowerDrawMW', offset=offset)
    smalldiff = pd.DataFrame({'0': traindf.ix[traindf['PowerDrawMW'] < 4
                                              , 'PowerDrawMWPred']
                                   - traindf.ix[traindf['PowerDrawMW'] < 4
                                                , 'PowerDrawMW'],
                              '1': traindf.ix[traindf['PowerDrawMW'] < 4
                                              , 'PowerDrawMW']})
    meddiff = pd.DataFrame({'0': traindf.ix[(traindf['PowerDrawMW'] >= 4)
                                            & (traindf['PowerDrawMW'] <= 15),
                                            'PowerDrawMWPred']
                                 - traindf.ix[(traindf['PowerDrawMW'] >= 4)
                                              & (traindf['PowerDrawMW'] <= 15),
                                              'PowerDrawMW'],
                            '1':traindf.ix[(traindf['PowerDrawMW'] >= 4)
                                           & (traindf['PowerDrawMW'] <= 15)
                                           , 'PowerDrawMW']})
    largediff = pd.DataFrame({'0' : traindf.ix[traindf['PowerDrawMW'] > 15
                                               , 'PowerDrawMWPred']
                                    - traindf.ix[traindf['PowerDrawMW'] > 15
                                                 , 'PowerDrawMW'],
                              '1': traindf.ix[traindf['PowerDrawMW'] > 15
                                              , 'PowerDrawMW']})
    smalldiff.dropna(inplace=True)
    meddiff.dropna(inplace=True)
    largediff.dropna(inplace=True)
    plt.hist(smalldiff.ix[:, 0].dropna(), 100, label='Power<4MW')
    plt.title('{} minute lag of PowerDrawMW < 4MW'.format(offset))
    plt.ylabel('{} min difference'.format(offset))
    plt.yscale('log')
    plt.savefig('histPowerDrawMW_{}minute_difference_lt4MW.png'.format(offset))
    plt.close()
    plt.hist(meddiff.ix[:, 0].dropna(), 100, label='4MW<=Power<=15')
    plt.title('{} minute lag of 4MW < PowerDrawMW < 15MW'.format(offset))
    plt.ylabel('{} min difference'.format(offset))
    plt.yscale('log')
    plt.savefig('histPowerDrawMW_{}minute_difference_4-15MW.png'.format(offset))
    plt.close()
    plt.hist(largediff.ix[:, 0], 100, label='Power>15MW')
    plt.title('{} minute lag of PowerDrawMW > 15MW'.format(offset))
    plt.ylabel('{} min difference'.format(offset))
    plt.yscale('log')
    plt.savefig('histPowerDrawMW_{}minute_difference_gt15MW.png'.format(offset))
    plt.close()
    bin_mean, bin_edges, bin_num = binned_statistic(smalldiff.ix[:, 1],
                                                    smalldiff.ix[:, 0],
                                                    statistic='mean',
                                                    bins=50)
    plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,
               label='binned statistic of data', zorder=2)
    plt.title('{} minute lag of 4MW < PowerDrawMW < 15MW'.format(offset))
    plt.ylabel('{} min difference'.format(offset))
    #plt.yscale('log')
    plt.savefig('hist2dPowerDrawMW_{}minute_difference_lt4MW.png'.format(offset))
    plt.close()
    bin_mean, bin_edges, bin_num = binned_statistic(meddiff.ix[:, 1],
                                                    meddiff.ix[:, 0],
                                                    statistic='mean',
                                                    bins=50)
    plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='b',
               lw=1, label='binned statistic of data', zorder=2)
    plt.title('{} minute lag of 4MW < PowerDrawMW < 15MW'.format(offset))
    plt.ylabel('{} min difference'.format(offset))
    #plt.yscale('log')
    plt.savefig('hist2dPowerDrawMW_{}minute_difference_4-15MW.png'.format(offset))
    plt.close()
    bin_mean, bin_edges, bin_num = binned_statistic(largediff.ix[:, 1],
                                                    largediff.ix[:, 0],
                                                    statistic='mean',
                                                    bins=50)
    plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:],
               colors='b', lw=1, label='binned statistic of data', zorder=2)
    plt.title('{} minute lag of 4MW < PowerDrawMW < 15MW'.format(offset))
    plt.savefig('hist2dPowerDrawMW_{}minute_difference_gt15MW.png'.format(offset))
    plt.close()


def dohistograms(df, dfcol):
    """
    Create simple histograms for exploratory analysis
    :param dfcol: dataframe column
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
    :param dfcol: dataframe column
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
    for i in sag['train'].columns.values:
        sag.dohistograms(sag['train'], i)

    for i in sag['train'].columns.values:
        #plt.close()
        sag.dotimeseries(sag['train'], i)
        #plt.close('all')


def doautocorrplot(sag):
    """
    plot autocorrelation
    :param df: the dataframe
    :param predictor: the predictor variable
    :return:
    """
    for i in sag.perfvars + sag.controlvars + sag.feedvars:
        plot_acf(sag['train'][i], lags=10000)
        plt.title('ACF {}'.format(i))
        plt.savefig('acf_{}.png'.format(i))
        plt.close('all')


def evalnn(sag, target='PowerDrawMW', offset=3, varlist=['Torque', 'PressA'],
           epochs=10, batch_size=4000, nvars=5):
    # evaluate model with standardized dataset
    """
    Evaluate keras neural network with cross_val_score
    :param sag: sag object
    :param modelfile: name of HDF5 file in case the model is saved
    :return: pipeline
    """
    traindata = sag.gettraindata(mode='train', targetvar=target,
                                 offset=offset)
    tvar = '{}Pred'.format(target)
    estimators = []
    #estimators.append(('standardize', StandardScaler()))
    estimators.append(('standardize', MinMaxScaler(feature_range=(-1, 1))))
    sagmodel = 'model{}_{}min'.format(target, offset)
    trainmodel = KerasRegressor(build_fn=SAGModels.medium_model, epochs=100,
                                optimizer='adam', init='normal', nvars=nvars,
                                batch_size=10000, verbose=0)
    #trainmodel = KerasRegressor(build_fn=SAGModels.two_var_medium_model,
    #                   epochs=100, optimizer='adam', init='normal',
    #                   batch_size=1000, verbose=0)
    estimators.append((sagmodel, trainmodel))
    X = traindata.drop(tvar, axis=1)
    Y = traindata[tvar]
###
    X = traindata.dropna().copy()
    #X = X.drop(sag.perfvars, axis=1)
    Y = X[tvar].copy()
    #X = X[varlist]
    X = X.drop(tvar, axis=1)
    X = X[[target]]
    #X = X.drop(target, axis=1)
    #X = X[['Torque','Speed']]
    Yval = Y.values.copy()
    yscaler = MinMaxScaler(feature_range=(-1, 1)).fit(Yval.reshape(-1, 1))
    #yscaler = StandardScaler().fit(Y)
    Yscaled = yscaler.transform(Yval.reshape(-1, 1))
###
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=5, random_state=SEED)
    #results = cross_val_score(pipeline, X, Yscaled, cv=kfold, n_jobs=-1)
    #print("results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    pipeline.fit(X, Yscaled)
    pipeline.yscaler = yscaler
    savemodel = True
    if savemodel:
        plfile = 'singlepipeline_{}.pkl'.format(sagmodel)
        modelfile = 'singlemodel_{}.h5'.format(sagmodel)
        pltemp = pipeline
        # Save model file first
        pltemp.named_steps[sagmodel].model.save(modelfile)
        pltemp.named_steps[sagmodel].model = None
        # add modelfile member to pipeline
        pltemp.modelfile = modelfile
        joblib.dump(pltemp, plfile)
        print 'Saved pipeline to {} and mode to {}'.format(plfile, modelfile)
    return pipeline


def nngrideval(sag, target='PowerDrawMW', offset=5,
               scaler=StandardScaler(), savemodel=False,
               savedir='', logfile=None):
    # evaluate model with standardized dataset
    """
    Evaluate Keras NN using GridSearchCV
    :param sag: sag object
    :param target: target variable
    :param offset: minute offset of prediction
    :param savemodel: saves pipeline to pkl file and keras model to h5 file
                      filename is generated from parameters
    :return: pipeline
    """
    traindata = sag.gettraindata(mode='train', targetvar=target, offset=offset)
    traindata = traindata.dropna()
    tvar = '{}Pred'.format(target)
    estimators = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    estimators.append(('standardize', scaler))
    sagmodel = 'model{}_{}min'.format(target, offset)
    #estimators.append((sagmodel, KerasRegressor(build_fn=larger_model,
    #estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.baseline_model,
    #estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.xlarge_model,
    #estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.two_var_medium_model,
    #estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.singlevar_model,
    estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.medium_model,
                                                epochs=150, optimizer='rmsprop',
                                                init='normal', batch_size=10000,
                                                verbose=0)))
    X = traindata.dropna().copy()
    Y = X[tvar].copy()
    X = X.drop(tvar, axis=1)
    #X = X.drop(target, axis=1)
    #X = traindata[['Torque','Speed']]
    Yval = Y.values.copy()
    #Y = Y.values.reshape(-1, 1)
    yscaler = MinMaxScaler(feature_range=(0, 1)).fit(Yval.reshape(-1, 1))
    #yscaler = StandardScaler().fit(Y)
    #Yscaled = yscaler.transform(Yval.reshape(-1,1))
    Yscaled = yscaler.transform(Y.reshape(-1, 1))
    pipeline = Pipeline(estimators)
    init = ['uniform']
    optimizers = ['rmsprop', 'adam']
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs = [50, 100, 150]
    batches = [10000, 50000, 100000]
    epochs = [50, 75, 100] #[150,256]
    #epochs = [100, 125, 150] #[150,256]
    epochs = [75, 100, 128] #[150,256]
    #epochs = [1,5,10,25]
    batches = [6000, 10000]
    optimizers = ['adam']
    init = ['normal']
    x1 = '{}__batch_size'.format(sagmodel)
    x2 = '{}__optimizer'.format(sagmodel)
    x3 = '{}__epochs'.format(sagmodel)
    x4 = '{}__init'.format(sagmodel)
    param_grid = {x1:batches, x2:optimizers, x3:epochs, x4:init}
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                        n_jobs=-1, scoring='neg_mean_squared_error')
    #X.index = range(X.shape[0])
    grid_result = grid.fit(np.array(X), np.array(Yscaled))
    #grid_result = grid.fit(np.array(X), np.array(Y))
    #grid_result = grid.fit(np.array(X), np.array(Y))
    #grid_result = grid.fit(X, Yscaled)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    print 'cv results', grid.cv_results_
    params = grid_result.cv_results_['params']
    bestest = grid_result.best_estimator_
    loglist = []
    loglist.append("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))
    for mean, stdev, param in zip(means, stds, params):
        loglist.append("%f (%f) with: %r\n" % (mean, stdev, param))
    print loglist
    if logfile is not None:
        with open(logfile, 'wb') as f:
            f.write(''.join(loglist))
    pipeline = bestest
    # Hack: add yscaler to pipeline for later use
    pipeline.yscaler = yscaler
    if savemodel:
        plfile = '{}skpipeline_{}.pkl'.format(savedir, sagmodel)
        modelfile = '{}skmodel_{}.h5'.format(savedir, sagmodel)
        pltemp = pipeline
        # Save model file first
        pltemp.named_steps[sagmodel].model.save(modelfile)
        pltemp.named_steps[sagmodel].model = None
        # add modelfile member to pipeline
        pltemp.modelfile = modelfile
        joblib.dump(pltemp, plfile)
        print 'Saved pipeline to {} and mode to {}'.format(plfile, modelfile)
    del traindata
    return pipeline


def sagpredict(sag, pipeline, mode='valid'):
    """
    Predict using pipeline with keras model
    :param sag: the sag object
    :param mode: the trained keras model
    :param return: target variable and DataFrame with predicted values
    """
    if type(pipeline) is str:
        pipeline = joblib.load(pipeline)
        laststep = pipeline.steps[-1][0]
        pipeline.named_steps[laststep].model = load_model(pipeline.modelfile)
    else:
        laststep = pipeline.steps[-1][0]
    targetvar, colname = laststep.split('_')
    print 'targetvar', targetvar
    idx1 = targetvar.find('linReg')
    #targetvar = targetvar.strip('model')
    if idx1 == -1:
        idx1 = targetvar.find('model')+len('model')
    else:
        idx1 = idx1 + len('linReg')
    targetvar = targetvar[idx1:]
    offset = int(colname.strip('min'))
    X = sag[mode].ix[:-offset, :]
    #X = X.drop(sag.controlvars, axis=1)
    #X = X.drop(sag.perfvars, axis=1)
    #X = X.drop(targetvar, axis=1)
    X = X[[targetvar]]
    #X = X[[target]+sag.feedvars+sag.controlvars]
    #X = X[sag.perfvars]
    yhat = pd.DataFrame(pipeline.predict(np.array(X)),
                        index=sag[mode].index[:-offset], columns=[colname])
    yhat.yscaler = pipeline.yscaler
    return targetvar, yhat


def singlevarnnpred(sag, pipelinelist, target='PowerDrawMW', mode='valid'):
    """
    Get 1-10minute prediction for variable <target>
    :param sag: the sag object
    :param pipelinelist: list with pipeline object, one for each minute
    :param target: target value
    :param mode: valid, test, train
    """
    dfyhat = pd.DataFrame(index=sag[mode].index,
                          columns=['1min', '2min', '3min', '4min',
                                   '5min', '6min', '7min', '8min',
                                   '9min', '10min'])
    dfyhat.yscalers = {}
    for i in pipelinelist:
        if type(i) is not str:
            laststep = i.steps[-1][0]
            targetvar, colname = laststep.split('_')
        else:
            targetvar = i.split('_')[1].strip('model')
        if targetvar.find(target) == -1:
            continue
        target, yhat = sagpredict(sag, i, mode=mode)
        dfyhat[yhat.columns[0]] = yhat
        dfyhat.yscalers[yhat.columns[0]] = yhat.yscaler
    return dfyhat


def runallsagpredict(sag, mode='valid', pipelinelist=None, model='linreg'):
    """
    Run Keras NN prediction on all performance variables
    and return dict with results
    :param sag: the sag object
    :param mode: run on train, valid or test data
    :param pipelinelist: list of pipeline. Can be either list of actual pipeline
                         objects or list of pipeline object files on disk
    :param model: underlying model to use. Can be either 'linreg' or 'keras'
    """
    if pipelinelist is None:
        if model not in ['linreg', 'keras']:
            print ('if no pipelinelist is passed, model argument has to be '
                   '\'linreg\' or \'keras\'')
    results = {}
    for i in sag.perfvars:
        df = pd.DataFrame(index=sag[mode].index,
                          columns=['1min', '2min', '3min', '4min',
                                   '5min', '6min', '7min', '8min',
                                   '9min', '10min'])
        df.yscalers = {}
        results[i] = df
    if pipelinelist is None:
        for j in sag.perfvars:
            for i in xrange(1, 11):
                if model == 'linreg':
                    pipeline = sag.linregforecast(targetvar=j, stepsahead=i)
                else:
                    pipeline = sag.nngrideval(target=j, offset=i,
                                              savemodel=False, logfile=None)
                target, yhat = sagpredict(sag, pipeline, mode=mode)
                colname = yhat.columns[0]
                print colname, j
                results[j].yscalers[colname] = yhat.yscaler
                results[j][colname] = yhat
    else:
        for i in pipelinelist:
            target, yhat = sagpredict(sag, i, mode=mode)
            colname = yhat.columns[0]
            results[target][colname] = yhat[colname]
            results[target].yscalers[colname] = yhat.yscaler
    return results


def fitlstm(sag, target='PowerDrawMW', offset=3, neurons=4, epochs=100,
            batch_size=200, savemodel=True):
    """
    Run LSTM fit on SAG training data
    """
    print 'In fitlstm'
    tvar = '{}Pred'.format(target)
    sagmodel = 'model{}_{}min'.format(target, offset)
    traindata = sag.gettraindata(mode='train', targetvar=target,
                                 offset=offset)
    X = traindata.dropna()
    Y = X[tvar].copy()
    X = X.drop(tvar, axis=1)
    X = X[[target]]
    # We'll use one timestep of a variable
    samples = X.shape[0]
    ntimesteps = 1
    features = X.columns.size
    Yval = Y.values.copy()
    yscaler = MinMaxScaler(feature_range=(0, 1)).fit(Yval.reshape(-1, 1))
    xscaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    yscaled = yscaler.transform(Yval.reshape(-1, 1))
    xscaled = xscaler.transform(X)
    xscaled = xscaled.reshape(samples, ntimesteps, features)
    yscaled = yscaled.reshape(samples, ntimesteps)
    trainmodel = SAGModels.lstm_model(batch_size, neurons,
                                      lookback=ntimesteps,
                                      nfeatures=xscaled.shape[2])
    for i in range(epochs):
        print 'fitting epoch {}'.format(i)
        trainmodel.fit(xscaled, yscaled, epochs=1, batch_size=batch_size,
                       verbose=0, shuffle=False)
        trainmodel.reset_states()
    print 'done fitting'
    minfo = ModelInfo(trainmodel, target, offset, neurons,
                      epochs, batch_size, xscaler, yscaler)
    if savemodel:
        # We have to save the keras model in a separate file
        modelfile = ('lstm_{}_{}min_{}neur_{}batch_{}ep.h5'.
                     format(target, offset, neurons, batch_size, epochs))
        minfo.modelname = modelfile
        trainmodel.save(modelfile)
        modelinfo = ('fit_{}_{}min_{}neur_{}batch_{}ep.pkl'.
                     format(target, offset, neurons, batch_size, epochs))
        joblib.dump(minfo, modelinfo)
        print 'Saved model info to {}'.format(modelinfo)
    return minfo


def fittenminutes(sag, target='Torque', save=True):
    """
    Run LSTM 1-10 minutes fits for target
    """
    minfolist = []
    for i in xrange(1, 11):
        minfo = fitlstm(sag, target, offset=i, savemodel=save)
        minfolist.append(minfo)
    return minfolist


def lstmpredict(sag, modelinfo, mode='train'):
    """
    Run prediction using info from modelinfo
    """
    lstmmodel = None
    if type(modelinfo) is str:
        modelinfo = joblib.load(modelinfo)
        lstmmodel = load_model(modelinfo.modelname)
    else:
        lstmmodel = modelinfo.modelname
    target = modelinfo.target
    X = sag[mode][:-modelinfo.offset]
    X = X[[target]]
    # We'll use one timestep of a variable
    samples = X.shape[0]
    ntimesteps = 1
    features = X.columns.size
    xscaled = modelinfo.xscaler.transform(X)
    xscaled = xscaled.reshape(samples, ntimesteps, features)
    print 'Running {} minute prediction for {}'.format(modelinfo.offset, target)
    yhatnorm = lstmmodel.predict(xscaled, batch_size=1)
    yhat = modelinfo.yscaler.inverse_transform(yhatnorm)
    return yhatnorm, yhat


def tenminuteforecast(sag, target='Torque', modellist=None,
                      mode='train', tocsv=False):
    """
    run 1-10 minute prediction with LSTM on target
    """
    columns = ['1min', '2min', '3min', '4min',
               '5min', '6min', '7min', '8min',
               '9min', '10min']
    yhatdf = pd.DataFrame(index=sag[mode].index,
                          columns=columns, dtype=np.float)
    # Add some meta data to the dataframe
    yhatdf.target = target
    for modelinfo in modellist:
        # Assume that target name is in file
        if type(modelinfo) is str:
            modelinfo = joblib.load(modelinfo)
            modelinfo.modelname = load_model(modelinfo.modelname)
        if modelinfo.target != target:
            print 'modelobject not for target {}'.format(target)
            continue
        tcol = '{}min'.format(modelinfo.offset)
        yhatnorm, yhat = lstmpredict(sag, modelinfo, mode)
        yhatdf[tcol].ix[:-modelinfo.offset] = yhat.ravel()
    if tocsv:
        csvout = 'lstmForecast_{}.csv'.format(target)
        yhatdf.to_csv('lstmForecast_{}_{}.csv'.format(target, mode))
        print 'Ten minute forecast for {} written to {}'.format(target, csvout)
    return yhatdf


def difference(dfdata, interval=1):
    """
    Create data with difference
    """
    diffdata = pd.DataFrame(index=dfdata.index, columns=dfdata.columns)
    for i in dfdata.columns:
        diffdata[i] = dfdata[i].diff(interval).dropna()
    return diffdata


def inverse_difference(diffdata, refdata, interval=1):
    """
    revert data difference
    """
    dforig = pd.DataFrame(index=diffdata.index, columns=diffdata.columns)
    dforig = (diffdata.shift(-interval) + refdata).shift(interval)
    dforig.ix[0] = refdata.ix[0]
    return dforig
