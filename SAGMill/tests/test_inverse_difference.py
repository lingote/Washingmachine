from __future__ import absolute_import
from unittest import TestCase
from pandas.util.testing import assert_frame_equal
from SAGMill.SAGData import SAGMillData
from SAGMill.SAGMillAnalysis import difference, inverse_difference

class TestInverse_difference(TestCase):
    def test_inverse_difference(self):
        sag = SAGMillData()
        diffdata = difference(sag.dfdata['train'])
        origdata = inverse_difference(diffdata,sag.dfdata['train'])
        assert_frame_equal(sag.dfdata['train'], origdata)