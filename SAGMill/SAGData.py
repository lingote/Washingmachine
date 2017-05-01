#-*- coding: utf-8 -*-
"""
Module for SAG mill analysis. This module loads SAG data and creates models
for prediction of various performance variables. Uses Keras with Theano backend.
"""
import pandas as pd
import numpy as np
from .SAGUtils import get_median_filtered


class SAGMillData(object):
    """
    Main class for SAG mill data pre-processing
    """
    def __init__(self, indata=('/run/media/ignacio/data/intellisense/'
                               'sag/SAG_data.csv'), teststart='2016-04-01'):
        """
        Return a new SAG object
        """
        self._indata = indata
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M')
        self.df = pd.read_csv(indata, parse_dates=['Time'], index_col='Time', date_parser=dateparse)
        self.df.drop('Unnamed: 0', axis=1, inplace=True)
        self.df.drop('Anomaly', axis=1, inplace=True)
        self.df = self.df.dropna()
        # Replace negative values with 0.
        self.df[self.df < 0] = 0.
        # Handle outliers
        outlieridx = []
        for j in self.df.columns:
            outlieridx += get_median_filtered(self.df[j], threshold=50).tolist()
        outlieridx = pd.DatetimeIndex(outlieridx, name='Time')
        self.df.drop(outlieridx, inplace=True)
        # rename columns, use shorter names, w/o spaces
        self.orignames = self.df.columns
        newcols = ['PressA', 'PressB', 'PressC', 'PressD', 'ConvBeltPSD', 'ConvBeltFines',
                   'ConvFeedRate', 'DilutionFlow', 'Torque', 'PowerDrawMW', 'SCATSConvBelt',
                   'Speed']
        self.df.columns = newcols
        self.dfdata = {}
        # Use last month of data for testing
        self.testindex = self.df[teststart:].index
        dataindex = self.df[:teststart].index
        self.trainindex = dataindex[:2*dataindex.shape[0]/3]
        self.validindex = dataindex[2*dataindex.shape[0]/3:]
        # define a scaler object
        self.controlvars = ['Speed (RPM)', 'Conveyor Belt Feed Rate (t/h)',
                            'Dilution Flow Rate (m3/h)']
        self.feedvars = ['Conveyor Belt PSD +4 (%)', 'Conveyor Belt PSD Fines (%)']
        #self.perfvars = ['Power Draw (MW)', 'Motor Torque (%)', 'Bearing Pressure A (kPa)',
        #            'Bearing Pressure B (kPa)', 'Bearing Pressure C (kPa)',
        #            'Bearing Pressure D (kPa)', 'SCATS Conveyor Belt Feed Rate (t/h)']
        self.perfvars = ['PressA', 'PressB', 'PressC', 'PressD',
                         'Torque', 'PowerDrawMW', 'SCATSConvBelt']
        self.feedvars = ['ConvBeltPSD', 'ConvBeltFines']
        self.controlvars = ['Speed', 'ConvFeedRate', 'DilutionFlow']


    def __getitem__(self, key):
        if key == 'test':
            return self.df[self.testindex]
        elif key == 'train':
            return self.df.ix[self.trainindex]
        elif key == 'valid':
            return self.df.ix[self.validindex]
        elif type(key) is list:
            if len(key) > 2:
                raise KeyError('key cannot have more than two values')
            elif len(key) == 1:
                return self.df.ix[key[0]]
            else:
                return self.df.ix[key[0]:key[1]]
        else:
            return self.df.ix[key]


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
        yfut = self['train'][targetvar].copy()
        yfut[:offset] = np.NaN
        yfut = yfut.shift(-offset)
        yfut.name = '{}Pred'.format(targetvar)
        traindata = self['train'].copy()
        traindata[yfut.name] = yfut
        return traindata


