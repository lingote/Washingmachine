#-*- coding: utf-8 -*-
"""
Some utils for SAG project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def get_median_filtered(signal, threshold=50):
    """
    Remove outliers. From 
    https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
    """
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return mask[mask==True].index


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

