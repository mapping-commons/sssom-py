"""Test for merging MappingSetDataFrames."""
import unittest

from sssom.io import extract_iri
from tests.constants import data_dir


class TestIO(unittest.TestCase):
    """A test case for merging msdfs."""

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
