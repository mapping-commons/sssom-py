"""Test for sorting MappingSetDataFrame columns."""

import unittest

from jsonschema import ValidationError
from linkml.validator.report import Severity, ValidationReport, ValidationResult

from sssom.constants import DEFAULT_VALIDATION_TYPES, SchemaValidationType
from sssom.parsers import parse_sssom_table
from sssom.validators import (
    SSSOMValidationError,
    format_validation_results,
    print_linkml_report,
    validate,
)
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

    def test_validate_reports_undeclared_prefix_with_hyphen(self) -> None:
        """Test that an undeclared prefix that contains a hyphen is reported."""
        msdf = parse_sssom_table(data_dir / "hyphen-prefix.sssom.tsv")
        validation_type = SchemaValidationType.PrefixMapCompleteness
        report = validate(msdf, [validation_type], fail_on_error=False)[validation_type]
        self.assertEqual(["Missing prefix: owasp-llm-2.0"], [r.message for r in report.results])

    def test_validate_json_rejects_non_uri_mapping_set_id(self) -> None:
        """Test that a mapping_set_id that is not a URI is rejected, and the error says why."""
        self.correct_msdf1.metadata["sssom_version"] = "1.1"
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        with self.assertRaises(ValidationError) as ctx:
            validate(self.correct_msdf1, [SchemaValidationType.JsonSchema])
        message = str(ctx.exception)
        self.assertIn("Your mapping set has 1 validation error:", message)
        self.assertIn("'not_a_uri' is not a 'uri' in /mapping_set_id", message)

    def test_validate_json_accepts_non_uri_mapping_set_id_in_sssom_1_0(self) -> None:
        """Test that a mapping_set_id that is not a URI is accepted in a SSSOM 1.0 set."""
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        rv = validate(self.correct_msdf1, [SchemaValidationType.JsonSchema])
        self.assertEqual([], rv[SchemaValidationType.JsonSchema].results)

    def test_validate_json_non_uri_mapping_set_id_report(self) -> None:
        """Test that a non-URI mapping_set_id is reported when not failing on error."""
        self.correct_msdf1.metadata["sssom_version"] = "1.1"
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        rv = validate(self.correct_msdf1, [SchemaValidationType.JsonSchema], fail_on_error=False)
        messages = [r.message for r in rv[SchemaValidationType.JsonSchema].results]
        self.assertEqual(["'not_a_uri' is not a 'uri' in /mapping_set_id"], messages)

    def test_validate_reports_problems_of_all_validation_types(self) -> None:
        """Test that a failing validation type does not hide the problems found by the others."""
        self.correct_msdf1.metadata["sssom_version"] = "1.1"
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        self.correct_msdf1.df.loc[0, "subject_id"] = "zzz:123"
        with self.assertRaises(SSSOMValidationError) as ctx:
            validate(self.correct_msdf1, self.validation_types)
        messages = [r.message for r in ctx.exception.results]
        self.assertIn("'not_a_uri' is not a 'uri' in /mapping_set_id", messages)
        self.assertIn("Missing prefix: zzz", messages)

    def test_validate_reports_non_curie_in_multivalued_metadata(self) -> None:
        """Test that a non-CURIE in a multivalued metadata slot is reported with other problems."""
        self.correct_msdf1.metadata["sssom_version"] = "1.1"
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        self.correct_msdf1.metadata["creator_id"] = ["orcid:1234", "nocolon"]
        with self.assertRaises(SSSOMValidationError) as ctx:
            validate(self.correct_msdf1, self.validation_types)
        messages = [r.message for r in ctx.exception.results]
        self.assertIn("'not_a_uri' is not a 'uri' in /mapping_set_id", messages)
        self.assertIn("Slot 'creator_id' has an incorrect value: nocolon", messages)

    def test_validate_without_failing_returns_report_per_type(self) -> None:
        """Test that each validation type gets its own report when not failing on error."""
        self.correct_msdf1.metadata["sssom_version"] = "1.1"
        self.correct_msdf1.metadata["mapping_set_id"] = "not_a_uri"
        self.correct_msdf1.df.loc[0, "subject_id"] = "zzz:123"
        rv = validate(self.correct_msdf1, self.validation_types, fail_on_error=False)
        self.assertEqual(
            ["'not_a_uri' is not a 'uri' in /mapping_set_id"],
            [r.message for r in rv[SchemaValidationType.JsonSchema].results],
        )
        self.assertEqual(
            ["Missing prefix: zzz"],
            [r.message for r in rv[SchemaValidationType.PrefixMapCompleteness].results],
        )

    def test_validate_inject_metadata_fixture(self) -> None:
        """Test that the test_inject_metadata_msdf.tsv fixture is a valid mapping set."""
        msdf = parse_sssom_table(data_dir / "test_inject_metadata_msdf.tsv")
        rv = validate(msdf, self.validation_types)
        self.assertEqual([], rv[SchemaValidationType.JsonSchema].results)

    def test_validate_reports_each_problem_once(self) -> None:
        """Test that a value that JSON Schema validation reports twice counts as one problem."""
        with self.assertRaises(SSSOMValidationError) as ctx:
            validate(self.bad_msdf1, self.validation_types)
        # 11 rows of bad_basic.tsv have a bad mapping_justification, which is reported once for
        # the pattern and once for the anyOf around it, and one prefix is missing.
        messages = [r.message for r in ctx.exception.results]
        self.assertEqual(12, len(messages))
        self.assertTrue(str(ctx.exception).startswith("Your mapping set has 12 validation errors:"))
        self.assertIn("Missing prefix: orcid", messages)

    def test_validate_without_failing_keeps_every_result(self) -> None:
        """Test that the returned reports keep every result, including repeated ones."""
        rv = validate(self.bad_msdf1, self.validation_types, fail_on_error=False)
        self.assertEqual(22, len(rv[SchemaValidationType.JsonSchema].results))


class TestPrintLinkMLReport(unittest.TestCase):
    """Test the formatting of validation reports."""

    @staticmethod
    def _report(n: int) -> ValidationReport:
        return ValidationReport(
            results=[
                ValidationResult(type="test", severity=Severity.ERROR, message=f"problem {i}")
                for i in range(n)
            ]
        )

    def test_valid_report_does_not_raise(self) -> None:
        """Test that an empty report does not raise."""
        print_linkml_report(self._report(0))

    def test_fail_on_error_false_does_not_raise(self) -> None:
        """Test that errors do not raise when fail_on_error is False."""
        print_linkml_report(self._report(3), fail_on_error=False)

    def test_error_message_lists_problems(self) -> None:
        """Test that the raised error lists every problem and uses the plural."""
        with self.assertRaises(ValidationError) as ctx:
            print_linkml_report(self._report(2))
        self.assertEqual(
            "Your mapping set has 2 validation errors:\n  - problem 0\n  - problem 1",
            str(ctx.exception),
        )

    def test_error_message_truncates_long_reports(self) -> None:
        """Test that the raised error caps the number of listed problems."""
        with self.assertRaises(SSSOMValidationError) as ctx:
            print_linkml_report(self._report(25))
        lines = str(ctx.exception).splitlines()
        self.assertEqual("Your mapping set has 25 validation errors:", lines[0])
        self.assertEqual("  - problem 19", lines[20])
        self.assertEqual("  ... and 5 more", lines[21])
        self.assertEqual(22, len(lines))
        self.assertEqual(25, len(ctx.exception.results))

    def test_only_fatal_and_error_raise(self) -> None:
        """Test that warnings and info are logged, and only FATAL and ERROR results raise."""
        for severity in (Severity.WARN, Severity.INFO):
            with self.subTest(severity=severity):
                report = ValidationReport(
                    results=[ValidationResult(type="test", severity=severity, message="fyi")]
                )
                print_linkml_report(report)
        for severity in (Severity.FATAL, Severity.ERROR):
            with self.subTest(severity=severity):
                report = ValidationReport(
                    results=[ValidationResult(type="test", severity=severity, message="broken")]
                )
                with self.assertRaises(SSSOMValidationError):
                    print_linkml_report(report)

    def test_error_is_a_jsonschema_validation_error(self) -> None:
        """Test that callers catching jsonschema's ValidationError keep working."""
        with self.assertRaises(ValidationError):
            print_linkml_report(self._report(1))

    def test_error_message_is_a_summary(self) -> None:
        """Test that the error's message attribute summarizes, and its string lists problems."""
        error = SSSOMValidationError(self._report(2).results)
        self.assertEqual("Your mapping set has 2 validation errors", error.message)
        self.assertIn("  - problem 1", str(error))

    def test_format_without_limit_lists_all_problems(self) -> None:
        """Test that all problems are listed when no limit is given."""
        text = format_validation_results(self._report(25).results)
        lines = text.splitlines()
        self.assertEqual(26, len(lines))
        self.assertEqual("  - problem 24", lines[-1])
