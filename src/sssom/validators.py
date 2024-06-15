"""Validators."""

import logging
from dataclasses import asdict
from typing import Callable, List, Mapping

from jsonschema import ValidationError
from linkml.validator import ValidationReport, Validator
from linkml.validator.plugins import JsonschemaValidationPlugin, RecommendedSlotsPlugin
from linkml.validator.report import Severity

from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_all_prefixes

from .constants import SCHEMA_YAML, SchemaValidationType


def validate(msdf: MappingSetDataFrame, validation_types: List[SchemaValidationType]) -> None:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    """
    for vt in validation_types:
        VALIDATION_METHODS[vt](msdf)


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
            if result.severity == Severity.ERROR:
                logging.error(result.message)

    if fail_on_error and validation_errors:
        raise ValidationError(f"You mapping set has {validation_errors} validation errors!")


# TODO This should not be necessary: https://github.com/linkml/linkml/issues/2117, https://github.com/orgs/linkml/discussions/1975
def clean_dict(d):
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
            cleaned_value = clean_dict(v)
            if cleaned_value:  # Add only if the nested dictionary is not empty
                cleaned_dict[k] = cleaned_value
        elif isinstance(v, list):
            # Recursively clean list elements if they are dictionaries
            cleaned_list = [clean_dict(item) if isinstance(item, dict) else item for item in v]
            # Filter out empty dictionaries from list
            cleaned_list = [item for item in cleaned_list if item not in [None, "", "null", {}]]
            if cleaned_list:  # Add only if the list is not empty
                cleaned_dict[k] = cleaned_list
        elif v not in [None, "", "null"]:
            cleaned_dict[k] = v

    return cleaned_dict


def validate_json_schema(msdf: MappingSetDataFrame) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    """
    validator = Validator(
        schema=SCHEMA_YAML,
        validation_plugins=[JsonschemaValidationPlugin(closed=False), RecommendedSlotsPlugin()],
    )
    mapping_set = to_mapping_set_document(msdf).mapping_set

    mapping_set_yaml = asdict(mapping_set)
    mapping_set_yaml_cleaned = clean_dict(mapping_set_yaml)

    report = validator.validate(mapping_set_yaml_cleaned, "mapping set")
    # TODO fail_on_error: False because of https://github.com/linkml/linkml/issues/2164
    print_linkml_report(report, False)


def validate_shacl(msdf: MappingSetDataFrame) -> None:
    """Validate SCHACL file.

    :param msdf: TODO: https://github.com/linkml/linkml/issues/850 .
    :raises NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def validate_sparql(msdf: MappingSetDataFrame) -> None:
    """Validate SPARQL file.

    :param msdf: MappingSetDataFrame
    :raises NotImplementedError: Not yet implemented.
    """
    # queries = {}
    # validator = SparqlDataValidator(SCHEMA_YAML,queries=queries)
    # mapping_set = to_mapping_set_document(msdf).mapping_set
    # TODO: Complete this function
    raise NotImplementedError


def check_all_prefixes_in_curie_map(msdf: MappingSetDataFrame) -> None:
    """Check all `EntityReference` slots are mentioned in 'curie_map'.

    :param msdf: MappingSetDataFrame
    :raises ValidationError: If all prefixes not in curie_map.
    """
    msdf.clean_context()
    missing_prefixes = get_all_prefixes(msdf).difference(msdf.converter.bimap)
    if missing_prefixes:
        raise ValidationError(f"The prefixes in {missing_prefixes} are missing from 'curie_map'.")


VALIDATION_METHODS: Mapping[SchemaValidationType, Callable] = {
    SchemaValidationType.JsonSchema: validate_json_schema,
    SchemaValidationType.Shacl: validate_shacl,
    SchemaValidationType.Sparql: validate_sparql,
    SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
}
