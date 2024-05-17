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
        self.bad_nando = parse_sssom_table(f"{data_dir}/mondo-nando.sssom.tsv")
        self.validation_types = DEFAULT_VALIDATION_TYPES
        self.shacl_validation_types = [SchemaValidationType.Shacl]

    def test_validate_json(self):
        """
        Test JSONSchemaValidation.

        Validate of the incoming file (basic.tsv) abides
        by the rules set by `sssom-schema`.
        """
        self.assertIsNone(validate(self.correct_msdf1, self.validation_types))

    @unittest.skip(
        reason="""\

    This test did not previously do what was expected. It was raising a validation error
    not because of the text below suggesting the validator was able to identify an issue
    with the `mapping_justification` slot, but because `orcid` was missing from the prefix map.
    The error actually thrown was::

      jsonschema.exceptions.ValidationError: The prefixes in {'orcid'} are missing from 'curie_map'.

    With updates in https://github.com/mapping-commons/sssom-py/pull/431, the default prefix map
    which includes `orcid` is added on parse, and this error goes away. Therefore, this test
    now fails, but again, this is a sporadic failure since the test was not correct in the first
    place. Therefore, this test is now skipped and marked for FIXME.
    """
    )
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

    def test_validate_nando(self):
        """Test Shacl validation (Not implemented)."""
        self.assertRaises(ValidationError, validate, self.bad_nando, self.validation_types)
