"""Tests for the default context."""

import unittest

from sssom.context import get_external_jsonld_context


class TestContext(unittest.TestCase):
    """A test case for the default context."""

    def test_minimum(self):
        """Test the minimum important prefixes are available through the default context."""
        external_context = get_external_jsonld_context()
        prefixes = set(external_context["@context"])
        expected_prefixes = {
            "FlyBase",
            "NCBITaxon",
            "NCBIGene",
            "NCBIProtein",
        }
        self.assertLessEqual(expected_prefixes, prefixes)
