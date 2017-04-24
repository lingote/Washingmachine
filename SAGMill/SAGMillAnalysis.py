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
from statsmodels.tsa.stattools import adfuller
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
import SAGModels


seed = 1234
np.random.seed(seed)

class SAGMillAnalyzer():
    """
    Main class for SAG mill performance prediction analysis
    """
    def __init__(self, indata='/run/media/ignacio/data/intellisense/sag/SAG_data.csv'):
        """
        Return a new SAG object
        """
        self._indata = indata
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
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
        # Use last month of data for testing
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


    def gettraindata(self, mode='train', targetvar='PowerDrawMW', offset=10):
        """
        Build dataset with target value <offset> minutes ahead
        :param sag: input sag object
        :param mode: train, valid or test data
        :param target: target variable
        :param offset: minute offset
        :return: pandasa dataframe with offset target variable
        """
        if mode not in ['test', 'train', 'valid']:
            print 'mode {} is not valid'.format(mode)
            return
        yfut = (pd.Series(self.dfdata[mode].loc[self.dfdata[mode].index[offset:],
                                               targetvar].values,
                          index=self.dfdata[mode]
                          .index[:-offset]))
        yfut.name = '{}Pred'.format(targetvar)
        return (pd.concat([self.dfdata[mode].loc[:self.dfdata[mode].index[-offset-1],
                                                :], yfut], copy=False, axis=1))


    def linregforecast(self, targetvar='PowerDrawMW', stepsahead=10, savemodel=False, scaler = StandardScaler()):
        """
        Create forecast for targetvar with 'stepsahead' of X
        :param self: input self object
        :param targetvar: target variable
        :param stepsahead: minutes ahead
        :return linregfit: fitted LinearRegression object
        """
        lr = linear_model.LinearRegression()
        targetname = '{}Pred'.format(targetvar)
        traindata = self.gettraindata(mode='train', targetvar=targetvar, offset=stepsahead)
        #linregfit = lr.fit(traindata.drop(targetname, axis=1), traindata[targetname])

        estimators = []
        estimators.append(('standardize', scaler))
        sagmodel = 'linReg{}_{}min'.format(targetvar, stepsahead)
        estimators.append((sagmodel, lr))
        X = traindata.drop(targetname, axis=1)
        Y = traindata[targetname]
        Y = Y.values.reshape(-1, 1)
        yscaler = StandardScaler().fit(Y)
        Yscale = yscaler.transform(Y)
        pipeline = Pipeline(estimators)
        pipeline.fit(X,Yscale)
        # Adding the yscaler to the pipeline
        pipeline.yscaler = yscaler
        if savemodel:
            plfile = 'skpipeline_{}.pkl'.format(sagmodel)
            # add modelfile member to pipeline
            joblib.dump(pipeline, plfile)
            print 'Saved pipeline to {}'.format(plfile)
        return pipeline


    def rsquared(self, results, mode='train'):
        """
        Get R^2 for all performance variable prediction for every
        1-10 minute forecast.
        :param self: input self object
        :param results: dict with perf variable as key, value is tuple of
                        fit results and prediction values
        :return: dictionary with forecast values for all performance variables
        """
        if mode not in ['test', 'train', 'valid']:
            print 'mode {} is not valid'.format(mode)
            return
        r2df = pd.DataFrame(index=self.perfvars)
        r2adjusteddf = pd.DataFrame(index=self.perfvars)
        for i in self.perfvars:
            r2 = []
            for j in results[i]:
                offset = int(j.strip('min'))
                yhat = results[i][j]
                yhat = yhat.loc[yhat.index[:-offset]]
                y = self.dfdata[mode].loc[self.dfdata[mode].index[offset:], i]
                sse = sum((y.values - yhat.values)**2)
                sstotal = sum((y - np.mean(y))**2)
                r_squared = 1 - (float(sse))/sstotal
                nregressor = self.dfdata[mode].shape[1]
                adjusted_r_squared = (r_squared - (1 - r_squared)*nregressor
                                      /(len(y.values) - nregressor - 1))
                r2df.loc[i, j] = r_squared
                r2adjusteddf.loc[i, j] = adjusted_r_squared
        return r2df, r2adjusteddf


    def calcresiduals(self, yhat, mode='valid', offset=1, target='PowerDrawMW'):
        """
        Compute difference between validation
        data and prediction at <offset> minutes.
        :return: Dataframe with observered y, and yerr
        """
        if mode not in ['test', 'train', 'valid']:
            print 'mode {} is not valid'.format(mode)
            return
        dfdata = self.dfdata[mode]
        vidx1 = dfdata.index[offset]
        vidx2 = dfdata.index[-1]
        # Create col name
        offsetcol = '{}min'.format(offset)
        diffpred = pd.DataFrame(index=self.dfdata[mode].index[offset:])
        diffpred['yobserved'] = self.dfdata[mode].loc[vidx1:vidx2, target].values
        yobs_scaled = diffpred['yobserved'].values
        yobs_scaled = yobs_scaled.reshape(-1, 1)
        yobs_scaled = yhat.yscalers[offsetcol].transform(yobs_scaled)
        yhat2 = yhat[offsetcol].dropna().values.reshape(-1,1)
        diffpred['yerr'] = (yobs_scaled - yhat2)
        return diffpred


    def drawsingleresiduals(self, yhats, offset=1, mode='valid', target='PowerDrawMW', modeldir='linreg'):
        """
        Draw residuals vs observed value. Save file in subdirectory <modeldir>
        :param self: sag object
        :param yhat: dict with yhat values, keys are perf variables
        :param mode: train, valid, test
        :param modeldir: save graphs in this sub-directory
        """
        residuals = self.calcresiduals(yhats, mode=mode, offset=offset, target=target)
        resmean = np.mean(residuals['yerr'])
        resstd = np.std(residuals['yerr'])
        bin_mean, bin_edges, bin_num = binned_statistic(residuals.ix[:,0], residuals.ix[:,1], statistic='mean', bins=50)
        bin_std, bin_stdedges, bin_stdnum = binned_statistic(residuals.ix[:,0], residuals.ix[:,1], statistic='std', bins=50)
        plt.scatter(residuals.ix[:,0], residuals.ix[:,1],zorder=1, s=1, color='gray')
        plt.hlines(bin_mean, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
        plt.hlines(bin_mean+bin_std, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
        plt.hlines(bin_mean-bin_std, bin_edges[:-1], bin_edges[1:], colors='b', lw=1,label='binned statistic of data',zorder=2)
        bin_mid = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2.
        plt.vlines(bin_mid, bin_mean-bin_std, bin_mean+bin_std, colors='b', lw=1,label='binned statistic of data',zorder=2)
        plt.title('{} {}min residuals'.format(target, offset))
        plt.xlabel('{}'.format(target))
        plt.ylabel('Residuals')
        plt.annotate('<r>={:6.3f}'.format(resmean), xy=(0.1, 0.9), xycoords='axes fraction')
        plt.annotate('std(r)={:6.3f}'.format(resstd), xy=(0.1, 0.8), xycoords='axes fraction')
        plt.savefig('{}/residual{}{}.png'.format(modeldir,target,offset))
        plt.close()


    def drawresiduals(self, yhats, target='PowerDrawMW', mode='valid', modeldir='linreg'):
        """
        Draw residuals vs observed value for 1-10min predictions for a single target variable.
        Save file in subdirectory <modeldir>
        :param self: self object
        :param yhats: dict with yhat values, keys are perf variables
        :param mode: train, valid, test
        :param modeldir: save graphs in this sub-directory
        """
        for j in range(1,11):
            self.drawsingleresiduals(yhats, offset=j, mode=mode, target=target, modeldir=modeldir)


    def drawallresiduals(self, yhats, mode='valid', modeldir='linreg'):
        """
        Draw residuals vs observed value. Save file in subdirectory <modeldir>
        :param sag: sag object
        :param yhat: dict with yhat values, keys are perf variables
        :param mode: train, valid, test
        :param modeldir: save graphs in this sub-directory
        """
        for i in self.perfvars:
            self.drawresiduals(yhats[i], target=i, mode=mode, modeldir=modeldir)


    def normalize(self, inverse=False):
        """
        Transform dataset using the scaler object attached to the SAG object
        :param inverse: True if revert normalization
        """
        colnames = self.dfdata['train'].columns
        trainindex = self.dfdata['train'].index
        validindex = self.dfdata['valid'].index
        testindex = self.dfdata['test'].index
        for i in self.dfdata.iterkeys():
            if inverse:
                self.dfdata[i] = (pd.DataFrame(self.scaler
                                              .inverse_transform(self.dfdata[i])
                                              , columns=colnames))
            else:
                self.dfdata[i] = (pd.DataFrame(self.scaler.transform(self.dfdata[i])
                                              , columns=colnames))
        self.dfdata['train'].index = trainindex
        self.dfdata['test'].index = testindex
        self.dfdata['valid'].index = validindex


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


    def savehists(self):
        """
        Create and save histograms and graphs of training data
        :return:
        """
        for i in self.dfdata['train'].columns.values:
            self.dohistograms(self.dfdata['train'], i)

        for i in self.dfdata['train'].columns.values:
            #plt.close()
            self.dotimeseries(self.dfdata['train'], i)
            #plt.close('all')


    def doautocorrplot(self, predictor, outname='ACFPlots.pdf'):
        """
        plot autocorrelation
        :param df: the dataframe
        :param predictor: the predictor variable
        :return:
        """
        with PdfPages(outname) as pp:
            for i in self.perfvars + self.controlvars + self.feedvars:
                plot_acf(self.dfdata['train'][i], lags=10000)
                plt.title('ACF {}'.format(i))
                pp.savefig()
                plt.close('all')


    def evalnn(self, target='PowerDrawMW', offset=5):
        # evaluate model with standardized dataset
        """
        Evaluate keras neural network with cross_val_score
        :param sag: sag object
        :param modelfile: name of HDF5 file in case the model is saved
        :return: pipeline
        """
        traindata = self.gettraindata(mode='train', targetvar=target, 
                                      offset=offset)
        tvar = '{}Pred'.format(target)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        sagmodel = 'model{}_{}min'.format(target, offset)
        trainmodel = KerasRegressor(build_fn=SAGModels.xlarge_model, epochs=50,
                                    optimizer='rmsprop', init='uniform',
                                    batch_size=100000, verbose=0)
        estimators.append((sagmodel, trainmodel))
        X = traindata.drop(tvar, axis=1)
        Y = traindata[tvar]
        pipeline = Pipeline(estimators)
        kfold = KFold(n_splits=5, random_state=seed)
        results = cross_val_score(pipeline, X, Y, cv=kfold)
        pipeline.fit(X,Y)
        Xstand = StandardScaler().fit_transform(X)
        yscaler = StandardScaler().fit(Y)
        trainmodel.fit(Xstand, yscaler.transform(Y))
        # Hack: add yscaler to pipeline for later use
        pipeline.yscaler = yscaler
        return pipeline


    def nngrideval(self, target='PowerDrawMW', offset=5, savemodel=False, savedir = '', logfile=None):
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
        traindata = self.gettraindata(mode='train', targetvar=target, offset=offset)
        tvar = '{}Pred'.format(target)
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        sagmodel = 'model{}_{}min'.format(target, offset)
        #estimators.append((sagmodel, KerasRegressor(build_fn=larger_model,
        #estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.baseline_model,
        estimators.append((sagmodel, KerasRegressor(build_fn=SAGModels.medium_model,
                           epochs=150, optimizer='rmsprop', init='normal',
                           batch_size=10000, verbose=0)))
        X = traindata.drop(tvar, axis=1)
        Y = traindata[tvar]
        Y = Y.values.reshape(-1, 1)
        yscaler = StandardScaler().fit(Y)
        Yscaled = yscaler.transform(Y)
        pipeline = Pipeline(estimators)
        init = ['uniform']
        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = [50, 100, 150]
        batches = [10000,50000,100000]
        epochs = [25, 50]
        epochs = [50]
        batches = [100000]
        optimizers = ['adam']
        init = ['normal']
        x1='{}__batch_size'.format(sagmodel)
        x2='{}__optimizer'.format(sagmodel)
        x3='{}__epochs'.format(sagmodel)
        x4='{}__init'.format(sagmodel)
        param_grid = {x1:batches, x2:optimizers, x3:epochs, x4:init}
        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=5)
        grid_result = grid.fit(X, Yscaled)
        #grid_result = grid.fit(X, Y)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
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
            plfile = '{}/skpipeline_{}.pkl'.format(savedir, sagmodel)
            modelfile = '{}/skmodel_{}.h5'.format(savedir, sagmodel)
            pltemp = pipeline
            # Save model file first
            pltemp.named_steps[sagmodel].model.save(modelfile)
            pltemp.named_steps[sagmodel].model = None
            # add modelfile member to pipeline
            pltemp.modelfile = modelfile
            joblib.dump(pltemp, plfile)
            print 'Saved pipeline to {} and mode to {}'.format(plfile, modelfile)
        return pipeline


    def sagpredict(self, pipeline, mode='valid'):
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
        targetvar = targetvar.strip('model')
        targetvar = targetvar.strip('linReg')
        offset = int(colname.strip('min'))
        X = self.dfdata[mode].ix[:-offset,:]
        yhat = pd.DataFrame(pipeline.predict(X),
                            index=self.dfdata[mode].index[:-offset], columns=[colname])
        yhat.yscaler = pipeline.yscaler
        return targetvar, yhat


    def singlevarnnpred(self, pipelinelist, target='PowerDrawMW', mode='valid'):
        """
        Get 1-10minute prediction for variable <target>
        :param sag: the sag object
        :param pipelinelist: list with pipeline object, one for each minute
        :param target: target value
        :param mode: valid, test, train
        """
        dfyhat = pd.DataFrame(index=self.dfdata[mode].index,
                              columns = ['1min', '2min', '3min', '4min',
                                         '5min', '6min', '7min', '8min',
                                         '9min', '10min'])
        dfyhat.yscalers = {}
        for i in pipelinelist:
            if type(i) is not str:
                laststep = i.steps[-1][0]
                targetvar, colname = laststep.split('_')
            else:
                targetvar = i.split('_')[1].strip('model')
            if targetvar.find(target)==-1:
                continue
            target, yhat = self.sagpredict(i, mode=mode)
            dfyhat[yhat.columns[0]] = yhat
            dfyhat.yscalers[yhat.columns[0]] = yhat.yscaler
        return dfyhat


    def runallsagpredict(self, mode='valid', pipelinelist=None, model='linreg'):
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
            if model not in ['linreg','keras']:
                print ('if no pipelinelist is passed, model argument has to be '
                       '\'linreg\' or \'keras\'')
        results = {}
        for i in self.perfvars:
            df = pd.DataFrame(index=self.dfdata[mode].index,
                              columns = ['1min', '2min', '3min', '4min',
                                         '5min', '6min', '7min', '8min',
                                         '9min', '10min'])
            df.yscalers = {}
            results[i] = df
        if pipelinelist is None:
            for j in self.perfvars:
                for i in xrange(1,11):
                   if model=='linreg':
                       pipeline = self.linregforecast(targetvar=j, stepsahead=i)
                   else:
                       pipeline = self.nngrideval(target=j, offset=i, savemodel=False, logfile=None)
                   target, yhat = self.sagpredict(pipeline, mode=mode)
                   colname = yhat.columns[0]
                   print colname, j
                   results[j].yscalers[colname] = yhat.yscaler
        else:
            for i in pipelinelist:
                target, yhat = self.sagpredict(i, mode=mode)
                colname = yhat.columns[0]
                results[target][colname] = yhat[colname]
                results[target].yscalers[colname] = yhat.yscaler
        return results


if __name__ == '__main__':
    sag = SAGMillAnalyzer()
    sag.savehists()
