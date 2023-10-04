"""Test for merging MappingSetDataFrames."""

import unittest

from sssom_schema import Mapping


class TestModel(unittest.TestCase):
    """A test case for making sure the model works as intended."""

    def test_invalid_mapping_throws_value_error(self):
        """Test if instantiating Mapping() fails when required elements are missing."""
        mdict_missing = dict(
            subject_id="ID:123"
        )  # This is missing object_id, predicate_id, mapping_justification

        with self.assertRaises(ValueError):
            _ = Mapping(**mdict_missing)

        with self.assertRaisesRegex(ValueError, "must be supplied"):
            _ = Mapping(**mdict_missing)

    def test_valid_mapping_does_not_throw_value_error(self):
        """Test if instantiating Mapping() works when all required elements are present."""
        mdict_complete = dict(
            subject_id="AID:123",
            object_id="BID:123",
            predicate_id="skos:exactMatch",
            mapping_justification="semapv:LexicalMatching",
        )
        m = Mapping(**mdict_complete)
        self.assertEqual(m.subject_id, "AID:123")
        self.assertEqual(m.object_id, "BID:123")
        self.assertEqual(m.predicate_id, "skos:exactMatch")
        self.assertEqual(m.mapping_justification, "semapv:LexicalMatching")
