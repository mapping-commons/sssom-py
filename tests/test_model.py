"""Test for merging MappingSetDataFrames."""

import unittest

from sssom_schema import Mapping


class TestModel(unittest.TestCase):
    """A test case for making sure the model works as intended."""

    def test_mapping_throws_value_error(self):
        """Test if instantiating Mapping() fails when required elements are missing."""
        mdict_missing = dict(
            subject_id="ID:123"
        )  # This is missing object_id, predicate_id, mapping_justification
        mdict_complete = dict(
            subject_id="AID:123",
            object_id="BID:123",
            predicate_id="skos:exactMatch",
            mapping_justification="semapv:LexicalMatching",
        )

        with self.assertRaises(ValueError):
            m = Mapping(**mdict_missing)

        m = Mapping(**mdict_complete)
        self.assertEqual(m.subject_id, "AID:123")
        self.assertEqual(m.object_id, "BID:123")
        self.assertEqual(m.predicate_id, "skos:exactMatch")
        self.assertEqual(m.mapping_justification, "semapv:LexicalMatching")
