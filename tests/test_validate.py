"""Test for sorting MappingSetDataFrame columns."""

import unittest

from jsonschema import ValidationError

from sssom.constants import DEFAULT_VALIDATION_TYPES, SchemaValidationType
from sssom.parsers import parse_sssom_table
from sssom.validators import validate
from tests.constants import data_dir


class TestValidate(unittest.TestCase):
    """A test case for sorting msdf columns."""

    def setUp(self) -> None:
        """Test up the test cases with the third basic example."""
        self.correct_msdf1 = parse_sssom_table(f"{data_dir}/basic.tsv")
        self.bad_msdf1 = parse_sssom_table(f"{data_dir}/bad_basic.tsv")
        self.validation_types = DEFAULT_VALIDATION_TYPES
        self.shacl_validation_types = [SchemaValidationType.Shacl]

    def test_validate_json(self):
        """
        Test JSONSchemaValidation.

        Validate of the incoming file (basic.tsv) abides
        by the rules set by `sssom-schema`.
        """
        results = validate(self.correct_msdf1, self.validation_types)
        self.assertEqual(len(results),2)

    def test_validate_json_fail(self):
        """
        Test if JSONSchemaValidation fail is as expected.

        In this particular test case, the 'mapping_justification' slot
        does not have EntityReference objects, but strings.
        """
        self.assertRaises(ValidationError, validate, self.bad_msdf1, self.validation_types)

    def test_validate_shacl(self):
        """
        Test Shacl validation (Not implemented).

        Validate shacl based on `sssom-schema`.
        """
        self.assertRaises(
            NotImplementedError,
            validate,
            self.correct_msdf1,
            self.shacl_validation_types,
        )

    def test_validate_sparql(self):
        """Test Shacl validation (Not implemented)."""
        self.assertRaises(
            NotImplementedError,
            validate,
            self.correct_msdf1,
            self.shacl_validation_types,
        )
