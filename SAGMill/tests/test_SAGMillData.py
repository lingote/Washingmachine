from __future__ import absolute_import
from unittest import TestCase, main
from SAGMill.SAGData import SAGMillData


class TestSAGMillData(TestCase):
    def test___getitem__(self):
        """
        train, valid, test indices should all be disjunct
        :return:
        """
        sag = SAGMillData()
        trainidx = set(sag['train'].index.values)
        valididx = set(sag['valid'].index.values)
        testidx = set(sag['test'].index.values)
        common = trainidx.intersection(valididx).intersection(testidx)
        self.assertEquals(len(common), 0)


    def test_gettraindata(self):
        sag = SAGMillData()
        for i in range(1, 11):
            dftest = sag.gettraindata(mode='train', targetvar='Torque', offset=i)
            sumdiff = (df.ix[dftest['TorquePred'].index,'Motor Torque (%)']
                       - dftest['TorquePred'].shift(i)).sum()
            self.assertEquals(sumdiff,0)
            numnull = pd.isnull(df.ix[dftest['TorquePred'].index,'Motor Torque (%)']
                                - dftest['TorquePred'].shift(i)).sum()
            self.assertEquals(numnull,i)


if __name__ == '__main__':
    main()