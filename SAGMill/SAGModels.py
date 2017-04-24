#-*- coding: utf-8 -*-
"""
Define some Neural Network models for Keras
"""
from keras.models import Sequential
import keras.models
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

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
    """
    A basic Neural Network
    """
    model = keras.models.Sequential()
    model.add(Dropout(dropout, input_shape=(13,)))
    model.add(Dense(13, input_dim=13, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def medium_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(13,)))
    model.add(Dense(18, input_dim=13, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(8, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def xlarge_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(13,)))
    #model.add(Dense(18, input_dim=13, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(24,  kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(18, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(6, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
