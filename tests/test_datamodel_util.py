import os
import unittest
from sssom.datamodel_util import MetaTSVConverter

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, 'data')


class GenerateSchemaTestCase(unittest.TestCase):
    def test_convert(self):
        cv = MetaTSVConverter()
        cv.load(f'{data_dir}/sssom_metadata.tsv')
        self.assertTrue(True)
        obj = cv.convert()
        print(obj)
        cv.convert_and_save(f'{data_dir}/sssom.yaml')



if __name__ == '__main__':
    unittest.main()
