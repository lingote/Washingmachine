#-*- coding: utf-8 -*-
"""
Define some Neural Network models for Keras
"""
from keras.models import Sequential
import keras.models
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import LSTM
from keras.constraints import maxnorm, non_neg


def singlevar_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    """
    A basic Neural Network using one var
    """
    model = keras.models.Sequential()
    model.add(Dense(1, input_dim=1, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def baseline_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    """
    A basic Neural Network
    """
    model = keras.models.Sequential()
    model.add(Dropout(dropout, input_shape=(12,)))
    model.add(Dense(12, input_dim=12, kernel_initializer=init, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def medium_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.3, nvars=8):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(nvars,)))
    model.add(Dense(11, input_dim=nvars, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(1)))
    model.add(Dropout(dropout))
    model.add(Dense(16, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(1)))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def medium2_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.25):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(12,)))
    model.add(Dense(32, input_dim=12, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(32, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def two_var_medium_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.3):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dense(32, input_dim=2, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def xlarge_model(optimizer='rmsprop', init='glorot_uniform', dropout=0.2):
    """
    A simple Neural Network
    """
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(12,)))
    #model.add(Dense(18, input_dim=12, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(16,  kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(16, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(16, kernel_initializer=init, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def lstm_model(batch_size, neurons, lookback, nfeatures):
    """
    LSTM model
    """
    dropout = 0.2
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, lookback, nfeatures), stateful=False, return_sequences=False))
    model.add(Dropout(dropout))
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, lookback , nfeatures), stateful=False, activation='tanh'))
    #model.add(LSTM(neurons, stateful=False, activation='tanh', return_sequences=True))
    #model.add(Dropout(dropout))
    #model.add(LSTM(neurons, batch_input_shape=(batch_size, lookback , nfeatures), stateful=False, activation='tanh'))
    #model.add(LSTM(neurons, stateful=False, activation='tanh'))
    #model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(dropout))
    #model.add(TimeDistributed(Dense(1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
