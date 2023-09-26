"""Test various grouping functionalities."""

import unittest

from sssom.parsers import parse_sssom_table
from sssom.util import (
    collapse,
    compare_dataframes,
    dataframe_to_ptable,
    filter_redundant_rows,
    group_mappings,
    parse,
    reconcile_prefix_and_data,
)
from tests.constants import data_dir


class TestCollapse(unittest.TestCase):
    """Test various grouping functionalities."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.df = parse(data_dir / "basic.tsv")

    def test_row_count(self):
        """Test the dataframe has the correct number of rows."""
        df = self.df
        self.assertEqual(
            len(df),
            141,
            f"Dataframe should have a different number of rows {df.head(10)}",
        )

    def test_collapse(self):
        """Test the row count after collapsing the dataframe."""
        df = collapse(self.df)
        self.assertEqual(len(df), 91, f"Dataframe should have a different {df.head(10)}")

    def test_filter(self):
        """Test the row count after filtering redundant rows."""
        df = filter_redundant_rows(self.df)
        self.assertEqual(len(df), 92)

    def test_ptable(self):
        """Test the row count of the ptable export."""
        rows = dataframe_to_ptable(self.df)
        self.assertEqual(91, len(rows))

    def test_groupings(self):
        """Test the row count after grouping mappings."""
        mappings = group_mappings(self.df)
        self.assertEqual(len(mappings), 91)

    def test_diff(self):
        """Test the comparison between two dataframes."""
        diff = compare_dataframes(self.df, self.df)
        self.assertEqual(0, len(diff.unique_tuples1))
        self.assertEqual(0, len(diff.unique_tuples2))
        self.assertEqual(91, len(diff.common_tuples))
        diff_df = diff.combined_dataframe
        self.assertLess(100, len(diff_df.index))
        for c in diff_df["comment"]:
            self.assertTrue(c.startswith("COMMON_TO_BOTH"))
        # output = sqldf("select * from diff_df where comment != ''")

        df2 = parse(data_dir / "basic2.tsv")
        diff = compare_dataframes(self.df, df2)
        self.assertEqual(15, len(diff.unique_tuples1))
        self.assertEqual(3, len(diff.unique_tuples2))
        self.assertEqual(76, len(diff.common_tuples))
        diff_df = diff.combined_dataframe

    def test_reconcile_prefix(self):
        """Test curie reconciliation is performing as expected."""
        msdf = parse_sssom_table(data_dir / "basic3.tsv")

        self.assertEqual(
            {
                "a": "http://example.org/a/",
                "b": "http://example.org/b/",
                "c": "http://example.org/c/",
                "d": "http://example.org/d/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "orcid": "https://orcid.org/",
                "semapv": "https://w3id.org/semapv/vocab/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "sssom": "https://w3id.org/sssom/",
            },
            msdf.prefix_map,
        )
        prefix_reconciliation = {
            "prefix_synonyms": {
                "a": "c",
                "c": "c2",
                "b": "bravo",
                "r": "rdfs",  # does not do anything, since "r" is not already in the prefix map
                "o": "owl",  # does not do anything, since "o" is not already in the prefix map
            },
            "prefix_expansion_reconciliation": {
                "c": "http://test.owl/c/",
                "bravo": "http://test.owl/bravo",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",  # matches what's already there
                "owl": "http://www.w3.org/2002/07/owl#",  # matches what's already there
            },
        }

        recon_msdf = reconcile_prefix_and_data(msdf, prefix_reconciliation)
        self.assertEqual(
            {
                "bravo": "http://test.owl/bravo",
                "c": "http://test.owl/c/",
                "c2": "http://example.org/c/",
                "d": "http://example.org/d/",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "orcid": "https://orcid.org/",
                "semapv": "https://w3id.org/semapv/vocab/",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "sssom": "https://w3id.org/sssom/",
            },
            recon_msdf.prefix_map,
        )
