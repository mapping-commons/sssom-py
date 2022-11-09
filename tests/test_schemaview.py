"""Test for sorting MappingSetDataFrame columns."""
import unittest

from sssom.constants import SSSOMSchemaView


class TestSort(unittest.TestCase):
    """A test case for sorting msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.sv = SSSOMSchemaView()

    def test_slots(self) -> None:
        """Test slots."""
        self.assertEqual(len(self.sv.mapping_slots), 39)
