# -*- coding: utf-8 -*-

"""Tests that important resources are available after build."""

import unittest

from sssom.constants import SCHEMA_YAML


class TestResources(unittest.TestCase):
    """A test case for resource availability checks."""

    def test_exists(self) -> None:
        """Test the schema YAML file is available to the package."""
        self.assertTrue(SCHEMA_YAML.is_file())
