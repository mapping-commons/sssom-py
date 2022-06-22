"""Test for sorting MappingSetDataFrame columns."""

import unittest

from sssom.constants import SCHEMA_DICT
from sssom.parsers import parse_sssom_table, to_mapping_set_document
from sssom.util import sort_df_rows_columns
from sssom.validators import json_schema_validate
from tests.constants import data_dir


class TestValidate(unittest.TestCase):
    """A test case for sorting msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.msdf = parse_sssom_table(f"{data_dir}/basic.tsv")

    def test_validate(self):
        """Test sorting of columns."""
        valid = json_schema_validate(self.msdf)
        self.assertTrue(valid)
