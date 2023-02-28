"""Test for merging MappingSetDataFrames."""
import unittest

from sssom.constants import OBJECT_ID, SUBJECT_ID
from sssom.io import extract_iri
from sssom.parsers import parse_sssom_table
from sssom.util import (
    MappingSetDataFrame,
    filter_out_prefixes,
    filter_prefixes,
    flip_mappings,
)
from tests.constants import data_dir


class TestIO(unittest.TestCase):
    """A test case for merging msdfs."""

    def setUp(self) -> None:
        """Set up."""
        self.msdf = parse_sssom_table(f"{data_dir}/basic.tsv")
        self.msdf2 = parse_sssom_table(f"{data_dir}/basic7.tsv")
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
        original_msdf = self.msdf
        filtered_df = filter_prefixes(
            original_msdf.df, prefix_filter_list, self.features
        )
        self.assertEqual(len(filtered_df), 40)

    def test_filter_out_prefixes(self):
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(
            original_msdf.df, prefix_filter_list, self.features
        )
        self.assertEqual(len(filtered_df), 5)

    def test_remove_mappings(self):
        """Test remove mappings."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(
            original_msdf.df, prefix_filter_list, self.features
        )
        new_msdf = MappingSetDataFrame(
            df=filtered_df,
            prefix_map=original_msdf.prefix_map,
            metadata=original_msdf.metadata,
        )
        original_length = len(original_msdf.df)
        original_msdf.remove_mappings(new_msdf)
        # len(self.msdf.df) = 141 and len(new_msdf.df) = 5
        self.assertEqual(len(original_msdf.df), original_length - len(new_msdf.df))

    def test_clean_prefix_map(self):
        """Test clean prefix map."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(
            original_msdf.df, prefix_filter_list, self.features
        )
        new_msdf = MappingSetDataFrame(
            df=filtered_df,
            prefix_map=original_msdf.prefix_map,
            metadata=original_msdf.metadata,
        )
        new_msdf.clean_prefix_map()
        self.assertEqual(
            new_msdf.prefix_map.keys(),
            set(original_msdf.prefix_map.keys()).intersection(
                set(new_msdf.prefix_map.keys())
            ),
        )

    def test_flip_nodes(self):
        """Test flip nodes."""
        subject_prefix = "a"
        original_msdf = self.msdf2
        flipped_df = flip_mappings(original_msdf.df, subject_prefix, False)
        self.assertEqual(len(flipped_df), 14)

    def test_flip_nodes_merged(self):
        """Test flip nodes."""
        subject_prefix = "a"
        original_msdf = self.msdf2
        flipped_df = flip_mappings(original_msdf.df, subject_prefix, True)
        self.assertEqual(len(flipped_df), 36)
