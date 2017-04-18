#-*- encoding: UTF-8 -*-
"""
Prototype module for SAG mill analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as sm
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict


class SAGMillAnalyzer():
    """
    Main class for our analysis
    """
    def __init__(self, indata = '/run/media/ignacio/data/intellisense/sag/SAG_data.csv'):
        self._indata = indata
        #self.df['Time'] = self.df['Time'].apply(lambda x: pd.to_datetime(x))
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
        #self.df = pd.read_csv(indata, usecols=range(1,15))
        self.df = pd.read_csv(indata, parse_dates=['Time'], index_col='Time',date_parser=dateparse)
        self.df.drop('Unnamed: 0',axis=1, inplace = True)
        self.df = self.df.dropna()
        # rename columns, use shorter names, w/o spaces
        self.orignames = self.df.columns
        newcols=['PressA', 'PressB', 'PressC', 'PressD', 'ConvBeltPSD', 'ConvBeltFines', 
                 'ConvFeedRate', 'DilutionFlow', 'Torque', 'PowerDrawMW', 'SCATSConvBelt', 
                 'Speed', 'Anomaly']
        self.df.columns = newcols
        self.test = self.df['2016-04-01':self.df.index[-1]] 
        self.train = self.df.drop(self.test.index)
        # take 2/3 for training, 1/3 for validation
        ntrain = self.train.shape[0]/3
        self.valid = self.train.loc[self.train.index[2*ntrain:]]
        self.train = self.train.loc[self.train.index[:2*ntrain]]
        # define a scaler object
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(self.train)
        self.controlvars = ['Speed (RPM)', 'Conveyor Belt Feed Rate (t/h)',
                      'Dilution Flow Rate (m3/h)']
        self.feedvars = ['Conveyor Belt PSD +4 (%)', 'Conveyor Belt PSD Fines (%)']
        self.perfvars = ['Power Draw (MW)', 'Motor Torque (%)', 'Bearing Pressure A (kPa)',
                    'Bearing Pressure B (kPa)', 'Bearing Pressure C (kPa)',
                    'Bearing Pressure D (kPa)', 'SCATS Conveyor Belt Feed Rate (t/h)']


def _createForecast(sag, targetvar = 'PowerDrawMW', stepsahead=10):
    """
    Create dataset with targetvar 'stepsahead' of X
    """
    yfut= pd.Series(sag.train.loc[sag.train.index[stepsahead:],targetvar].values, index=sag.train.index[:-stepsahead])
    yfut.name = '{}Pred'.format(targetvar)
    fitdata = pd.concat([sag.train.loc[:sag.train.index[-stepsahead-1],:], yfut],axis=1, copy=False)
    formula = ("{}Pred ~ Speed + ConvFeedRate + DilutionFlow + Torque + PressA + "
               "PressB + PressC + PressD + ConvBeltPSD + ConvBeltFines + "
               "ConvFeedRate + SCATSConvBelt").format(targetvar)
    return sm.ols(formula=formula, data = fitdata).fit()


def createForecast(sag, targetvar = 'PowerDrawMW', stepsahead=10):
    """
    Create dataset with targetvar 'stepsahead' of X
    """
    lr = linear_model.LinearRegression()
    yfut= pd.Series(sag.train.loc[sag.train.index[stepsahead:],targetvar].values, index=sag.train.index[:-stepsahead])
    yfut.name = '{}Pred'.format(targetvar)
    linregfit = lr.fit(sag.train.loc[:sag.train.index[-stepsahead-1],:], yfut)
    return linregfit


def doTenMinuteForecast(sag, target='PowerDrawMW'):
    fits = []
    for i in xrange(1,11):
        fits.append(_createForecast(sag, targetvar = target, stepsahead = i))
    valpred = pd.concat([pd.DataFrame(i.predict(sag.valid), index = sag.valid.index) for i in fits], axis=1)
    valpred.columns = ['{}min'.format(i) for i in range(1,11)]
    return fits, valpred


def getPredError(sag, pred, target = 'PowerDrawMW'):
    """
    Calculate prediction error. Compute difference between prediction
    and validation data.
    """
    diffpred = []
    for i in range(pred.index.shape[0]-10):
        tenMinPred = pred.loc[pred.index[i]]
        validvals = sag.valid.loc[sag.valid.index[i+1:i+11],target]
        diffpred.append(tenMinPred.values - validvals.values)
    return diffpred


def getPredErrorMin(sag, pred, offset = 1, target = 'PowerDrawMW'):
    """
    Calculate prediction error. Compute difference between validation 
    data and prediction at <offset> minutes.
    """
    pidx1 = pred.index[0]
    pidx2 = pred.index[-(1+offset)]
    vidx1 = sag.valid.index[offset]
    vidx2 = sag.valid.index[-1]
    # Create col name
    offsetcol = '{}min'.format(offset)
    diffpred = sag.valid.loc[vidx1:vidx2, target].values - pred.loc[pidx1:pidx2, offsetcol].values
    return diffpred


def plotPredVsDataMin(sag, pred, offset = 1, target = 'PowerDrawMW'):
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
    ax.plot(sag.valid.loc[vidx1:vidx2, target].values, pred.loc[pidx1:pidx2, offsetcol].values, 'o')
    plt.title("{}: Predicted vs data (validation)".format(target))
    plt.xlabel("Data {}".format(target))
    plt.ylabel("Predicted {}".format(target))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_xlim([sag.valid.loc[vidx1:vidx2, target].min(),sag.valid.loc[vidx1:vidx2, target].max()])
    plt.show()

    
def normalize(sag, inverse = False):
    """
    Transform dataset using the scaler object attached to the SAG object
    """
    colnames = sag.train.columns
    trainindex = sag.train.index
    validindex = sag.valid.index
    testindex = sag.test.index
    if inverse:
        sag.train = pd.DataFrame(sag.scaler.inverse_transform(sag.train),columns = colnames)
        sag.valid = pd.DataFrame(sag.scaler.inverse_transform(sag.valid), columns = colnames)
        sag.test = pd.DataFrame(sag.scaler.inverse_transform(sag.test), columns = colnames)
    else:
        sag.train = pd.DataFrame(sag.scaler.transform(sag.train), columns = colnames)
        sag.valid = pd.DataFrame(sag.scaler.transform(sag.valid), columns = colnames)
        sag.test = pd.DataFrame(sag.scaler.transform(sag.test), columns = colnames)
    sag.train.index = trainindex
    sag.test.index = testindex
    sag.valid.index = validindex


def dohistograms(df, dfcol, pp):
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
    pp.savefig()


def dotimeseries(df, dfcol, pp = None):
    plt.close('all')
    ts = df[dfcol]
    plt.xlabel('time')
    plt.ylabel(dfcol)
    plt.title('TimeSeries of {}'.format(dfcol))
    plt.plot(ts)
    pp.savefig()


def savehists(sag, histfile = None, tsfile = None):
    """
    Save histograms to PDF file
    :return:
    """
    if histfile is None and tsfile is None:
        print("enter at either hist or ts output filename")
        return

    if tsfile is not None:
        with PdfPages(histfile) as pp:
            for i in sag.perfvars + sag.controlvars + sag.feedvars:
                h1 = dohistograms(sag.train, i, pp)

    if tsfile is not None:
        with PdfPages(tsfile) as pp:
            for i in sag.perfvars + sag.controlvars + sag.feedvars:
                plt.close()
                t1 = dotimeseries(sag.train, i, pp)
        plt.close('all')


def doautocorrplot(sag, predictor, outname='ACFPlots.pdf'):
    """
    plot autocorrelation
    :param df: the dataframe
    :param predictor: the predictor variable
    :return:
    """
    with PdfPages(outname) as pp:
        for i in sag.perfvars + sag.controlvars + sag.feedvars:
            plot_acf(sag.df[i], lags = 10000)
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
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


if __name__ == '__main__':
    sag = SAGMillAnalyzer()
    sag.savehists()
