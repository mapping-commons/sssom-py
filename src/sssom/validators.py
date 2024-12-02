"""Validators."""

import logging
from typing import Callable, List, Mapping

from jsonschema import ValidationError
from linkml.validator import ValidationReport, Validator
from linkml.validator.plugins import JsonschemaValidationPlugin
from linkml.validator.report import Severity, ValidationResult
from linkml_runtime.dumpers import json_dumper

from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_all_prefixes

from .constants import SCHEMA_YAML, SchemaValidationType, _get_sssom_schema_object


def validate(
    msdf: MappingSetDataFrame,
    validation_types: List[SchemaValidationType],
    fail_on_error: bool = True,
) -> None:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    :param fail_on_error: If true, throw an error when execution of a method has failed
    """
    for vt in validation_types:
        VALIDATION_METHODS[vt](msdf, fail_on_error)


def print_linkml_report(report: ValidationReport, fail_on_error: bool = True):
    """Print the error messages in the report. Optionally throw exception.

    :param report: A LinkML validation report
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
    """
    validation_errors = 0

    if not report.results:
        logging.info("The instance is valid!")
    else:
        for result in report.results:
            validation_errors += 1
            if (result.severity == Severity.FATAL) or (result.severity == Severity.ERROR):
                logging.error(result.message)
            elif result.severity == Severity.WARN:
                logging.error(result.message)
            elif result.severity == Severity.INFO:
                logging.info(result.message)

    if fail_on_error and validation_errors:
        raise ValidationError(f"You mapping set has {validation_errors} validation errors!")


# TODO This should not be necessary: https://github.com/linkml/linkml/issues/2117,
#  https://github.com/orgs/linkml/discussions/1975
def _clean_dict(d):
    """Recursively removes key-value pairs from a dictionary where the value is None, "null", or an empty string.

    Args:
    d (dict): The dictionary to clean.

    Returns:
    dict: A cleaned dictionary with unwanted values removed.
    """
    if not isinstance(d, dict):
        return d

    cleaned_dict = {}
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


def validate_json_schema(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
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
    print_linkml_report(report, fail_on_error)


def validate_shacl(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> None:
    """Validate SCHACL file.

    :param msdf: TODO: https://github.com/linkml/linkml/issues/850 .
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
    :raises NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def validate_sparql(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> None:
    """Validate SPARQL file.

    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
    :raises NotImplementedError: Not yet implemented.
    """
    # queries = {}
    # validator = SparqlDataValidator(SCHEMA_YAML,queries=queries)
    # mapping_set = to_mapping_set_document(msdf).mapping_set
    # TODO: Complete this function
    raise NotImplementedError


def check_all_prefixes_in_curie_map(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> None:
    """Check all `EntityReference` slots are mentioned in 'curie_map'.

    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
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
    print_linkml_report(report, fail_on_error)


def check_strict_curie_format(msdf: MappingSetDataFrame, fail_on_error: bool = True) -> None:
    """Check all `EntityReference` slots are formatted as unambiguous curies.

    Implemented rules:
      - CURIE does not contain pipe "|" character to ensure that multivalued processing of in TSV works correctly.
    :param msdf: MappingSetDataFrame
    :param fail_on_error: if true, the function will throw an ValidationError exception when there are errors
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
            for index, value in msdf.df[column].items():
                if pd.notna(value) and "|" in str(value):
                    message = f"{value} contains a pipe ('|') character (row {index + 1}, column '{column}')."
                    validation_results.append(
                        ValidationResult(
                            type="strict curie test",
                            severity=Severity.ERROR,
                            message=message,
                        )
                    )

    report = ValidationReport(results=validation_results)
    print_linkml_report(report, fail_on_error)


VALIDATION_METHODS: Mapping[SchemaValidationType, Callable] = {
    SchemaValidationType.JsonSchema: validate_json_schema,
    SchemaValidationType.Shacl: validate_shacl,
    SchemaValidationType.Sparql: validate_sparql,
    SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    SchemaValidationType.StrictCurieFormat: check_strict_curie_format,
}
