"""Tests for the default context."""

import unittest

from sssom.context import get_converter


class TestContext(unittest.TestCase):
    """A test case for the default context."""

    def test_minimum(self):
        """Test the minimum important prefixes are available through the default context."""
        prefixes = get_converter().get_prefixes()
        expected_prefixes = {
            "FlyBase",
            "NCBITaxon",
            "NCBIGene",
            "NCBIProtein",
            "oboInOwl",
        }
        self.assertLessEqual(expected_prefixes, prefixes)
