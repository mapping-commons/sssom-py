"""Validators."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Mapping, Optional

from jsonschema import ValidationError
from linkml.validator import ValidationReport, Validator
from linkml.validator.plugins import JsonschemaValidationPlugin
from linkml.validator.report import Severity, ValidationResult
from linkml_runtime.dumpers import json_dumper

from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_all_prefixes, get_prefix_from_curie

from .constants import (
    DEFAULT_VALIDATION_TYPES,
    SCHEMA_YAML,
    SchemaValidationType,
    _get_sssom_schema_object,
)


def validate(
    msdf: MappingSetDataFrame,
    validation_types: Optional[List[SchemaValidationType]] = None,
    fail_on_error: bool = True,
) -> dict[SchemaValidationType, ValidationReport]:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    All requested validation types are run before anything is raised, so that the problems found
    by all of them are reported together. Nothing is logged; use :func:`print_linkml_report` or
    :func:`format_validation_results` to show the returned reports.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    :param fail_on_error: If true, raise an error when any validation type reports problems

    :returns: A dictionary from validation types to validation reports

    :raises SSSOMValidationError: If fail_on_error is true and any validation type reports
        problems.
    """
    if validation_types is None:
        validation_types = DEFAULT_VALIDATION_TYPES
    reports = {vt: _REPORT_METHODS[vt](msdf) for vt in validation_types}
    if fail_on_error:
        results = [result for report in reports.values() for result in report.results]
        if results:
            raise SSSOMValidationError(results)
    return reports


#: Maximum number of problems listed when an SSSOMValidationError is converted to a string
MAX_REPORTED_PROBLEMS = 20

_LOG_LEVELS = {
    Severity.FATAL: logging.CRITICAL,
    Severity.ERROR: logging.ERROR,
    Severity.WARN: logging.WARNING,
    Severity.INFO: logging.INFO,
}


def _summarize(results: List[ValidationResult]) -> str:
    noun = "error" if len(results) == 1 else "errors"
    return f"Your mapping set has {len(results)} validation {noun}"


def format_validation_results(results: List[ValidationResult], limit: Optional[int] = None) -> str:
    """Format validation results as a human-readable list of problems.

    :param results: The validation results to format.
    :param limit: The maximum number of problems to list, or None to list all of them.

    :returns: A summary line followed by one line per problem.
    """
    lines = [f"{_summarize(results)}:"]
    shown = results if limit is None else results[:limit]
    lines.extend(f"  - {r.message}" for r in shown)
    if len(results) > len(shown):
        lines.append(f"  ... and {len(results) - len(shown)} more")
    return "\n".join(lines)


class SSSOMValidationError(ValidationError):
    """Raised when a mapping set fails validation.

    All problems are available in :attr:`results`. The string representation lists at most
    :data:`MAX_REPORTED_PROBLEMS` of them, so that tracebacks stay readable.
    """

    def __init__(self, results: List[ValidationResult]) -> None:
        """Instantiate the error from the validation results that caused it."""
        super().__init__(_summarize(results))
        self.results = results

    def __str__(self) -> str:
        """List the problems, up to :data:`MAX_REPORTED_PROBLEMS` of them."""
        return format_validation_results(self.results, limit=MAX_REPORTED_PROBLEMS)


def print_linkml_report(report: ValidationReport, fail_on_error: bool = True) -> None:
    """Print the error messages in the report. Optionally throw exception.

    :param report: A LinkML validation report
    :param fail_on_error: if true, the function will throw an SSSOMValidationError exception when
        there are errors, listing the problems in its message. Otherwise, the problems are logged.

    :raises SSSOMValidationError: If fail_on_error is true and the report has results.
    """
    results = report.results
    if not results:
        logging.info("The instance is valid!")
        return

    if fail_on_error:
        raise SSSOMValidationError(results)

    for result in results:
        logging.log(_LOG_LEVELS.get(result.severity, logging.ERROR), result.message)


# TODO This should not be necessary: https://github.com/linkml/linkml/issues/2117,
#  https://github.com/orgs/linkml/discussions/1975
def _clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively removes key-value pairs from a dictionary where the value is None, "null", or an empty string.

    :param d: The dictionary to clean.

    :returns: A cleaned dictionary with unwanted values removed.
    """
    if not isinstance(d, dict):
        return d

    cleaned_dict: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively clean nested dictionary
            cleaned_value = _clean_dict(v)
            if cleaned_value:  # Add only if the nested dictionary is not empty
                cleaned_dict[k] = cleaned_value
        elif isinstance(v, list):
            # Recursively clean list elements if they are dictionaries
            cleaned_list = [_clean_dict(item) if isinstance(item, dict) else item for item in v]
            # Filter out empty dictionaries from list
            cleaned_list = [item for item in cleaned_list if item not in [None, "", "null", {}]]
            if cleaned_list:  # Add only if the list is not empty
                cleaned_dict[k] = cleaned_list
        elif v not in [None, "", "null"]:
            cleaned_dict[k] = v

    return cleaned_dict


def validate_json_schema(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> ValidationReport:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    :param fail_on_error: if true, the function will throw an ValidationError exception when there
        are errors
    """
    report = _json_schema_report(msdf)
    print_linkml_report(report, fail_on_error)
    return report


def _json_schema_report(msdf: MappingSetDataFrame) -> ValidationReport:
    validator = Validator(
        schema=SCHEMA_YAML,
        validation_plugins=[JsonschemaValidationPlugin(closed=False)],
    )
    mapping_set = to_mapping_set_document(msdf).mapping_set

    # mapping_set_yaml = asdict(mapping_set)
    # mapping_set_dict = _clean_dict(mapping_set_yaml)
    mapping_set_dict = json_dumper.to_dict(mapping_set)

    return validator.validate(mapping_set_dict, "mapping set")


def validate_shacl(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> ValidationReport:
    """Validate SCHACL file.

    :param msdf: TODO: https://github.com/linkml/linkml/issues/850 .
    :param fail_on_error: if true, the function will throw an ValidationError exception when there
        are errors

    :raises NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def validate_sparql(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> ValidationReport:
    """Validate SPARQL file.

    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there
        are errors

    :raises NotImplementedError: Not yet implemented.
    """
    # queries = {}
    # validator = SparqlDataValidator(SCHEMA_YAML,queries=queries)
    # mapping_set = to_mapping_set_document(msdf).mapping_set
    # TODO: Complete this function
    raise NotImplementedError


def check_all_prefixes_in_curie_map(
    msdf: MappingSetDataFrame, fail_on_error: bool = True
) -> ValidationReport:
    """Check all `EntityReference` slots are mentioned in 'curie_map'.

    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there
        are errors

    :raises ValidationError: If all prefixes not in curie_map.
    """
    report = _prefix_map_completeness_report(msdf)
    print_linkml_report(report, fail_on_error)
    return report


def _prefix_map_completeness_report(msdf: MappingSetDataFrame) -> ValidationReport:
    msdf.clean_context()
    missing_prefixes = get_all_prefixes(msdf).difference(msdf.converter.bimap)
    validation_results = []
    for prefix in missing_prefixes:
        validation_results.append(
            ValidationResult(
                type="prefix validation",
                severity=Severity.ERROR,
                instance=None,
                instantiates=None,
                message=f"Missing prefix: {prefix}",
            )
        )
    entity_reference_slots = _get_sssom_schema_object().entity_reference_slots
    for slot, value in msdf.metadata.items():
        if slot not in entity_reference_slots or not isinstance(value, list):
            continue
        for curie in value:
            if not get_prefix_from_curie(curie):
                validation_results.append(
                    ValidationResult(
                        type="prefix validation",
                        severity=Severity.ERROR,
                        message=f"Slot '{slot}' has an incorrect value: {curie}",
                    )
                )
    return ValidationReport(results=validation_results)


def check_strict_curie_format(
    msdf: MappingSetDataFrame, fail_on_error: bool = True
) -> ValidationReport:
    """Check all `EntityReference` slots are formatted as unambiguous curies.

    Implemented rules:

    1. CURIE does not contain pipe "|" character to ensure that multivalued processing of in TSV
       works correctly.

    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there
        are errors

    :raises ValidationError: If any entity reference does not follow the strict CURIE format
    """
    report = _strict_curie_format_report(msdf)
    print_linkml_report(report, fail_on_error)
    return report


def _strict_curie_format_report(msdf: MappingSetDataFrame) -> ValidationReport:
    import itertools as itt

    import pandas as pd

    msdf.clean_context()
    validation_results = []
    metadata_keys = set(msdf.metadata.keys())
    entity_reference_slots = {
        slot
        for slot in itt.chain(metadata_keys, msdf.df.columns.to_list())
        if slot in _get_sssom_schema_object().entity_reference_slots
        if slot not in _get_sssom_schema_object().multivalued_slots
    }

    for column in entity_reference_slots:
        if column in msdf.df.columns:
            for index, value in enumerate(msdf.df[column], start=1):
                if pd.notna(value) and "|" in str(value):
                    message = (
                        f"{value} contains a pipe ('|') character (row {index}, column '{column}')."
                    )
                    validation_results.append(
                        ValidationResult(
                            type="strict curie test",
                            severity=Severity.ERROR,
                            message=message,
                        )
                    )

    return ValidationReport(results=validation_results)


def _not_implemented_report(msdf: MappingSetDataFrame) -> ValidationReport:
    raise NotImplementedError


VALIDATION_METHODS: Mapping[SchemaValidationType, Callable[..., ValidationReport]] = {
    SchemaValidationType.JsonSchema: validate_json_schema,
    SchemaValidationType.Shacl: validate_shacl,
    SchemaValidationType.Sparql: validate_sparql,
    SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    SchemaValidationType.StrictCurieFormat: check_strict_curie_format,
}

#: Functions that build the report of each validation type without logging or raising
_REPORT_METHODS: Mapping[
    SchemaValidationType, Callable[[MappingSetDataFrame], ValidationReport]
] = {
    SchemaValidationType.JsonSchema: _json_schema_report,
    SchemaValidationType.Shacl: _not_implemented_report,
    SchemaValidationType.Sparql: _not_implemented_report,
    SchemaValidationType.PrefixMapCompleteness: _prefix_map_completeness_report,
    SchemaValidationType.StrictCurieFormat: _strict_curie_format_report,
}
