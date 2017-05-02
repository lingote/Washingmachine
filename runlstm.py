#-*- coding: utf-8 -*-
"""
Simple function to run forecast for 1-10 minutes of a given
SAG mill performance variable
"""
import argparse
import timeit
import SAGMill.SAGMillAnalysis
import SAGMill.SAGData


fitdict = {'PressA': 'pressa_fits.txt', 'PressB': 'pressb_fits.txt',
           'PressC': 'pressc_fits.txt', 'PressD': 'pressd_fits.txt',
           'Torque': 'torque_fits.txt', 'PowerDrawMW': 'powerdraw_fits.txt',
           'SCATSConvBelt': 'scatsconvbelt_fits.txt'}


def main():
    """
    main function calls writer function
    :param argv: args list
    :return:
    """
    perfvars = ['PressA', 'PressB', 'PressC', 'PressD',
                'Torque', 'PowerDrawMW', 'SCATSConvBelt']
    parser = argparse.ArgumentParser(description=("Run SAG prediction for performance variable"))
    parser.add_argument('-t', '--target-var', type=str, required=True,
                        choices=perfvars, dest='t', help='target variable')
    parser.add_argument('-s', '--save-forecast', type=bool, required=False,
                        default=False, dest='s', help='save forecast to csv')
    parser.add_argument('-f', '--fits', type=bool, required=False,
                        default=False, dest='f', help='run fits or only prediction')
    parser.add_argument('-m', '--mode', type=str, required=False,
                        default='valid', dest='m', help='train, valid or test mode')
    parser.add_argument('-d', '--draw', type=bool, required=False,
                        default=False, dest='m', help='draw residuals')
    args = parser.parse_args()
    sag = SAGMill.SAGData.SAGMillData()
    if args.f:
        fits = SAGMill.SAGMillAnalysis.fittenminutes(sag, target=args.t)
    else:
        with open(fitdict[args.t],'rb') as f:
            fits = f.readlines()
            fits = [i.strip('\n') for i in fits]
    dfyhat = SAGMill.SAGMillAnalysis.tenminuteforecast(sag, target=args.t,
                                                       modellist=fits,
                                                       mode=args.m, tocsv=args.s)
    if args.d:
        SAGMill.SAGMillAnalysis.drawresiduals(sag, dfyhat, target=args.t,
                                              mode=args.m, modeldir='lstm')


if __name__ == '__main__':
    main()
