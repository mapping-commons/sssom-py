"""Test for built-in prefixes."""

import unittest

from sssom.context import get_extended_prefix_map, get_jsonld_context


class TestPrefix(unittest.TestCase):
    """A test case for testing prefixes using EPM."""

    def test_builtin_prefixes(self):
        """This test ensures that the bioregistry managed EPM (extended prefix-map) does not deviate from the fixed SSSOM built-in prefixes."""
        prefix_map = get_extended_prefix_map()
        sssom_schema_context = get_jsonld_context()
        for k, v in prefix_map.items():
            if isinstance(v, str):
                if k in sssom_schema_context["@context"]:
                    self.assertTrue(sssom_schema_context["@context"][k] == v)
