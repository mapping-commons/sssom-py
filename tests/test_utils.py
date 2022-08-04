"""Test for merging MappingSetDataFrames."""
import unittest

from sssom.constants import OBJECT_ID, SUBJECT_ID
from sssom.io import extract_iri
from sssom.parsers import parse_sssom_table
from sssom.util import MappingSetDataFrame, filter_out_prefixes, filter_prefixes
from tests.constants import data_dir


class TestIO(unittest.TestCase):
    """A test case for merging msdfs."""

    def setUp(self) -> None:
        """Set up."""
        self.msdf = parse_sssom_table(f"{data_dir}/basic.tsv")
        self.features = [SUBJECT_ID, OBJECT_ID]

    def test_broken_predicate_list(self):
        """Test merging of multiple msdfs."""
        pred_filter_list = ["skos:relatedMatch", f"{data_dir}/predicate_list3.txt"]
        prefix_map = {"skos": "http://www.w3.org/2004/02/skos/core#"}
        iri_list = []
        for p in pred_filter_list:
            p_iri = extract_iri(p, prefix_map)
            if p_iri:
                iri_list.extend(p_iri)
        self.assertEqual(3, len(iri_list))

    def test_filter_prefixes(self):
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        filtered_df = filter_prefixes(self.msdf.df, prefix_filter_list, self.features)
        self.assertEqual(len(filtered_df), 40)

    def test_filter_out_prefixes(self):
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        filtered_df = filter_out_prefixes(
            self.msdf.df, prefix_filter_list, self.features
        )
        self.assertEqual(len(filtered_df), 5)

    def test_remove_mappings(self):
        """Test remove mappings."""
        prefix_filter_list = ["x", "y"]
        filtered_df = filter_out_prefixes(
            self.msdf.df, prefix_filter_list, self.features
        )
        new_msdf = MappingSetDataFrame(
            df=filtered_df, prefix_map=self.msdf.prefix_map, metadata=self.msdf.metadata
        )
        original_length = len(self.msdf.df)
        self.msdf.remove_mappings(new_msdf)
        # len(self.msdf.df) = 141 and len(new_msdf.df) = 5
        self.assertEqual(len(self.msdf.df), original_length - len(new_msdf.df))
