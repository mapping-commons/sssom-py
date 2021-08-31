import os
import unittest

from sssom.parsers import read_sssom_table, read_sssom_json, read_sssom_rdf
from sssom.writers import write_table, write_json, write_rdf, write_owl

# from pandasql import sqldf
from tests.test_data import test_out_dir, test_data_dir

cwd = os.path.abspath(os.path.dirname(__file__))


class TestWrite(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists(test_out_dir):
            os.mkdir(test_out_dir)
        self.msdf = read_sssom_table(f"{test_data_dir}/basic.tsv")
        # self.msdf = read_sssom_table(f"{test_data_dir}/basic-simple.tsv")
        self.mapping_count = 141  # 141 for basic.tsv

    def test_write_sssom_dataframe(self):
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_dataframe.tsv")
        write_table(self.msdf, tmp_file)
        msdf = read_sssom_table(tmp_file)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{tmp_file} has the wrong number of mappings.",
        )

    def test_write_sssom_rdf(self):
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_rdf.rdf")
        write_rdf(self.msdf, tmp_file)
        msdf = read_sssom_rdf(tmp_file, self.msdf.prefixmap)
        write_table(
            self.msdf, os.path.join(test_out_dir, "test_write_sssom_rdf.rdf.tsv")
        )
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{tmp_file} has the wrong number of mappings.",
        )

    def test_write_sssom_json(self):
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_json.json")
        write_json(self.msdf, tmp_file)
        msdf = read_sssom_json(tmp_file)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{tmp_file} has the wrong number of mappings.",
        )

    def test_write_sssom_owl(self):
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_owl.owl")
        write_owl(self.msdf, tmp_file)
        self.assertEqual(1, 1)
