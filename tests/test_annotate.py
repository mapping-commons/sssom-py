"""Test for annotating MappingSetDataFrame metadata."""

import unittest
from os.path import join

from sssom.io import annotate_file
from sssom.parsers import parse_sssom_table
from tests.constants import data_dir


class TestSort(unittest.TestCase):
    """A test case for filtering msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.input = join(data_dir, "basic.tsv")

        self.validation_file = join(data_dir, "test_annotate_sssom.tsv")

    def test_annotate(self):
        """Test annotation of metadata."""
        kwargs = {
            "mapping_set_id": ("http://w3id.org/my/mapping.sssom.tsv",),
            "mapping_set_version": ("2021-01-01",),
        }
        annotated_msdf = annotate_file(input=self.input, **kwargs)
        validation_msdf = parse_sssom_table(self.validation_file)

        self.assertEqual(annotated_msdf.metadata, validation_msdf.metadata)
        self.assertEqual(annotated_msdf.prefix_map, validation_msdf.prefix_map)
        self.assertEqual(len(annotated_msdf.df), len(validation_msdf.df))

    def test_annotate_multivalued(self):
        """Test annotation of metadata which are multivalued."""
        kwargs = {
            "creator_id": ("orcid:0123",),
        }
        annotated_msdf = annotate_file(input=self.input, **kwargs)

        self.assertTrue(len(annotated_msdf.metadata["creator_id"]), 3)

        # Pass same ORCID.
        kwargs = {
            "creator_id": ("orcid:1234",),
        }
        annotated_msdf_2 = annotate_file(input=self.input, **kwargs)
        self.assertTrue(len(annotated_msdf_2.metadata["creator_id"]), 2)

    def test_annotate_fail(self):
        """Pass invalid param to see if it fails."""
        kwargs = {"abcd": ("x:%", "y:%")}
        with self.assertRaises(ValueError):
            annotate_file(input=self.input, **kwargs)
