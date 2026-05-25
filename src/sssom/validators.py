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
from sssom.util import MappingSetDataFrame, get_all_prefixes

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

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    :param fail_on_error: If true, throw an error when execution of a method has failed

    :returns: A dictionary from validation types to validation reports
    """
    if validation_types is None:
        validation_types = DEFAULT_VALIDATION_TYPES
    return {vt: VALIDATION_METHODS[vt](msdf, fail_on_error) for vt in validation_types}


def format_report(report: ValidationReport, label: str = "") -> str:
    """Format a ValidationReport as one line per unique JSON path, dropping the instance blob."""
    prefix = f"[{label}] " if label else ""
    lines: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for r in report.results:
        src = r.source
        if src is not None:
            path = tuple(src.absolute_path)
            if path in seen:
                continue
            seen.add(path)
            path_str = "/".join(str(p) for p in path) or "<root>"
            lines.append(f"{prefix}{path_str}: {src.message}")
        else:
            lines.append(f"{prefix}{r.message}")
    return "\n".join(lines)


_FAILING_SEVERITIES = (Severity.FATAL, Severity.ERROR)


def _raise_if_errors(report: ValidationReport, fail_on_error: bool) -> None:
    """Raise a ValidationError summarising ERROR/FATAL results, if ``fail_on_error``."""
    n = sum(1 for r in report.results if r.severity in _FAILING_SEVERITIES)
    if fail_on_error and n:
        raise ValidationError(f"You mapping set has {n} validation errors!")


def print_linkml_report(report: ValidationReport, fail_on_error: bool = True) -> None:
    """Log each result in the report and optionally raise on errors.

    Retained for callers that want logging-style output; the built-in validators no longer
    invoke it themselves so that callers can choose their own rendering (e.g. ``format_report``).
    """
    for result in report.results:
        if result.severity in (Severity.FATAL, Severity.ERROR, Severity.WARN):
            logging.error(result.message)
        else:
            logging.info(result.message)
    if not report.results:
        logging.info("The instance is valid!")
    _raise_if_errors(report, fail_on_error)


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
    validator = Validator(
        schema=SCHEMA_YAML,
        validation_plugins=[JsonschemaValidationPlugin(closed=False)],
    )
    mapping_set = to_mapping_set_document(msdf).mapping_set

    # mapping_set_yaml = asdict(mapping_set)
    # mapping_set_dict = _clean_dict(mapping_set_yaml)
    mapping_set_dict = json_dumper.to_dict(mapping_set)

    report = validator.validate(mapping_set_dict, "mapping set")
    _raise_if_errors(report, fail_on_error)
    return report


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
    report = ValidationReport(results=validation_results)
    _raise_if_errors(report, fail_on_error)
    return report


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

    report = ValidationReport(results=validation_results)
    _raise_if_errors(report, fail_on_error)
    return report


VALIDATION_METHODS: Mapping[SchemaValidationType, Callable[..., ValidationReport]] = {
    SchemaValidationType.JsonSchema: validate_json_schema,
    SchemaValidationType.Shacl: validate_shacl,
    SchemaValidationType.Sparql: validate_sparql,
    SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    SchemaValidationType.StrictCurieFormat: check_strict_curie_format,
}
