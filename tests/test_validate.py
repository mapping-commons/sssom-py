"""Test for sorting MappingSetDataFrame columns."""

import unittest

from jsonschema import ValidationError
from linkml.validator.report import Severity, ValidationReport, ValidationResult

from sssom.constants import DEFAULT_VALIDATION_TYPES, SchemaValidationType
from sssom.parsers import parse_sssom_table
from sssom.validators import _raise_if_errors, format_report, validate
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

    def test_format_report_bad(self) -> None:
        """format_report dedupes per JSON path and emits non-empty output for a bad file."""
        reports = validate(self.bad_msdf1, self.validation_types, fail_on_error=False)
        json_lines = format_report(
            reports[SchemaValidationType.JsonSchema], label="JsonSchema"
        ).splitlines()
        # 11 bad mapping_justification rows in bad_basic.tsv; deduped by path
        self.assertEqual(11, len(json_lines))
        self.assertTrue(all(line.startswith("[JsonSchema] mappings/") for line in json_lines))
        prefix_out = format_report(
            reports[SchemaValidationType.PrefixMapCompleteness], label="PrefixMapCompleteness"
        )
        self.assertEqual("[PrefixMapCompleteness] Missing prefix: orcid", prefix_out)

    def test_format_report_clean(self) -> None:
        """format_report returns the empty string when there are no results."""
        reports = validate(self.correct_msdf1, self.validation_types, fail_on_error=False)
        for report in reports.values():
            self.assertEqual("", format_report(report))

    def test_raise_if_errors_respects_severity(self) -> None:
        """_raise_if_errors raises only on FATAL/ERROR, never on WARN/INFO."""
        info = ValidationReport(
            results=[ValidationResult(type="t", severity=Severity.INFO, message="just fyi")]
        )
        warn = ValidationReport(
            results=[ValidationResult(type="t", severity=Severity.WARN, message="watch out")]
        )
        error = ValidationReport(
            results=[ValidationResult(type="t", severity=Severity.ERROR, message="broken")]
        )
        _raise_if_errors(info, fail_on_error=True)  # no raise
        _raise_if_errors(warn, fail_on_error=True)  # no raise
        self.assertRaises(ValidationError, _raise_if_errors, error, True)
