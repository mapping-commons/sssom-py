"""Test for sorting MappingSetDataFrame columns."""

import io
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

    def test_validate_json(self) -> None:
        """Test JSONSchemaValidation.

        Validate of the incoming file (basic.tsv) abides by the rules set by `sssom-schema`.
        """
        rv = validate(self.correct_msdf1, self.validation_types)
        self.assertIsNotNone(rv)
        self.assertIn(SchemaValidationType.JsonSchema, rv)
        json_validation = rv[SchemaValidationType.JsonSchema]
        self.assertEqual([], json_validation.results)

    @unittest.skip(reason="""\

    This test did not previously do what was expected. It was raising a validation error
    not because of the text below suggesting the validator was able to identify an issue
    with the `mapping_justification` slot, but because `orcid` was missing from the prefix map.
    The error actually thrown was::

      jsonschema.exceptions.ValidationError: The prefixes in {'orcid'} are missing from 'curie_map'.

    With updates in https://github.com/mapping-commons/sssom-py/pull/431, the default prefix map
    which includes `orcid` is added on parse, and this error goes away. Therefore, this test
    now fails, but again, this is a sporadic failure since the test was not correct in the first
    place. Therefore, this test is now skipped and marked for FIXME.
    """)
    def test_validate_json_fail(self) -> None:
        """Test if JSONSchemaValidation fail is as expected.

        In this particular test case, the 'mapping_justification' slot does not have EntityReference
        objects, but strings.
        """
        self.assertRaises(ValidationError, validate, self.bad_msdf1, self.validation_types)

    def test_validate_shacl(self) -> None:
        """Test Shacl validation (Not implemented).

        Validate shacl based on `sssom-schema`.
        """
        self.assertRaises(
            NotImplementedError,
            validate,
            self.correct_msdf1,
            self.shacl_validation_types,
        )

    def test_validate_sparql(self) -> None:
        """Test Shacl validation (Not implemented)."""
        self.assertRaises(
            NotImplementedError,
            validate,
            self.correct_msdf1,
            self.shacl_validation_types,
        )

    def test_validate_nando(self) -> None:
        """Test Shacl validation (Not implemented)."""
        self.assertRaises(ValidationError, validate, self.bad_nando, self.validation_types)

    def test_validate_prefix_map_completeness(self) -> None:
        """Test that prefixes containing a hyphen or following a pipe must be declared."""
        msdf = parse_sssom_table(f"{data_dir}/hyphen-and-pipe-prefixes.sssom.tsv")
        validation_type = SchemaValidationType.PrefixMapCompleteness
        report = validate(msdf, [validation_type], fail_on_error=False)[validation_type]
        self.assertEqual(
            {"Missing prefix: my-vocab", "Missing prefix: other"},
            {r.message for r in report.results},
        )

    def test_validate_prefix_map_completeness_set_and_records(self) -> None:
        """Test that a slot given on the set and on its records is read in both places."""
        stream = io.StringIO(
            "# curie_map:\n"
            "#   ex: http://example.org/ex/\n"
            "# mapping_set_id: https://example.org/sets/set-and-records\n"
            "# license: https://creativecommons.org/publicdomain/zero/1.0/\n"
            "# subject_source: ex:source\n"
            "subject_id\tpredicate_id\tobject_id\tmapping_justification\tsubject_source\n"
            "ex:1\tskos:exactMatch\tex:2\tsemapv:ManualMappingCuration\tother:source\n"
        )
        msdf = parse_sssom_table(stream)
        validation_type = SchemaValidationType.PrefixMapCompleteness
        report = validate(msdf, [validation_type], fail_on_error=False)[validation_type]
        self.assertEqual({"Missing prefix: other"}, {r.message for r in report.results})
