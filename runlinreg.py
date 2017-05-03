#-*- coding: utf-8 -*-
"""
Simple function to run forecast using linear regression
for 1-10 minutes of a given SAG mill performance variable
"""
import argparse
import SAGMill.SAGMillAnalysis
import SAGMill.SAGData


def main():
    """
    main function runs linear regression and creates residual graphs
    :return:
    """
    parser = argparse.ArgumentParser(description=("Run SAG prediction with linear regression"))
    parser.add_argument('-o', '--output-dir', type=str, required=False,
                        default='linreg', dest='out', help='output dir')
    args = parser.parse_args()
    sag = SAGMill.SAGData.SAGMillData()
    linres = SAGMill.SAGMillAnalysis.runallsagpredict(sag)
    SAGMill.SAGMillAnalysis.drawallresiduals(sag, linres, modeldir=args.out)


if __name__=='__main__':
    main()
