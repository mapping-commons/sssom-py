import os
import unittest
from sssom.datamodel_util import MetaTSVConverter


class MetaTestCase(unittest.TestCase):
    def test_convert(self):
        cv = MetaTSVConverter()
        cv.dirname = 'tests/kbase'
        cv.load('data/sssom_metadata.tsv')
        self.assertTrue(True)
        obj = cv.convert()
        print(obj)
        cv.convert_and_save('data/sssom.yaml')



if __name__ == '__main__':
    unittest.main()
