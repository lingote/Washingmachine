#-*- coding: utf-8 -*-
"""
Simple function to draw residuals of 
SAG mill performance prediction
"""
import argparse
import pandas as pd
import SAGMill.SAGMillAnalysis
import SAGMill.SAGData


def main():
    """
    main function calls writer function
    :param argv: args list
    :return:
    """
    perfvars = ['PressA', 'PressB', 'PressC', 'PressD',
                'Torque', 'PowerDrawMW', 'SCATSConvBelt']
    parser = argparse.ArgumentParser(description=("Create plots SAG prediction residuals"))
    parser.add_argument('-t', '--target-var', type=str, required=True,
                        choices=perfvars, dest='t')
    parser.add_argument('-f', '--file-name', type=str, required=True,
                        default=False, dest='f')
    parser.add_argument('-m', '--mode', type=str, required=False,
                        default='valid', dest='m')
    args = parser.parse_args()
    sag = SAGMill.SAGData.SAGMillData()
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    dfyhat = pd.read_csv(args.f, parse_dates=['Time'], index_col='Time', date_parser=dateparse)
    # Need to add some metadata to yhat
    # Assuming nameing convention a la:
    # lstmForecast_PowerDrawMW_4neurons_100ep_200batch_train.csv
    idx = args.f.find('neurons')
    idx2 = args.f.rfind('_',0,idx)
    dfyhat.neurons = args.f[idx2+1:idx]
    idx = args.f.find('ep_')
    idx2 = args.f.rfind('_',0,idx)
    dfyhat.epochs = args.f[idx2+1:idx]
    idx = args.f.find('batch_')
    idx2 = args.f.rfind('_',0,idx)
    dfyhat.batch = args.f[idx2+1:idx]
    SAGMill.SAGMillAnalysis.drawresiduals(sag, dfyhat, target=args.t,
                                          mode=args.m, modeldir='lstm')


if __name__ == '__main__':
    main()
