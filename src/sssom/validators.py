"""Validators."""

import logging
from typing import Dict, List

from jsonschema import ValidationError

# from linkml.validators.jsonschemavalidator import JsonSchemaDataValidator
# from linkml.validators.sparqlvalidator import SparqlDataValidator  # noqa: F401
from linkml_runtime.processing.referencevalidator import ReferenceValidator
from linkml_runtime.utils.schemaview import SchemaView
from sssom_schema import MappingSet

from sssom.context import add_built_in_prefixes_to_prefix_map
from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_all_prefixes

from .constants import SCHEMA_YAML, SchemaValidationType


class ValidationResult:
    """An individual validation result from a validation system."""

    def __init__(self, category: str):
        """Initialise the validation result.

        :param category: Name of the category this validation result belongs to.
        """
        self.message = ""
        self.category = category
        self.check = ""
        self.data: Dict = {}


class ValidationReport:
    """A validation report that contains the results of various validation methods."""

    def __init__(self, name: str):
        """Initialise the validation report.

        :param name: Name of the category this validation report corresponds to.
        """
        self.name = name
        self.results: List[ValidationResult] = []

    def add_result(self, result: ValidationResult):
        """Add result to the validation report."""
        self.results.append(result)

    def print_report(self):
        """Print the all results of the report."""
        print(f"Validation Report: {self.name}")
        for result in self.results:
            print(f"Category: {result.category}, Message: {result.message}")


def validate(
    msdf: MappingSetDataFrame, validation_types: List[SchemaValidationType]
) -> List[ValidationReport]:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    """
    validation_methods = {
        SchemaValidationType.JsonSchema: validate_json_schema,
        SchemaValidationType.Shacl: validate_shacl,
        SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    }
    results = []
    for vt in validation_types:
        result = validation_methods[vt](msdf)
        results.append(result)
    return results


def validate_json_schema(msdf: MappingSetDataFrame, fail_hard=False) -> ValidationReport:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    :param fail_hard:
    """
    validator = ReferenceValidator(SchemaView(SCHEMA_YAML))
    mapping_set = to_mapping_set_document(msdf).mapping_set
    results = validator.validate(mapping_set, MappingSet)
    validation_results = ValidationReport("sssom_linkml_reference_validator")
    if fail_hard and results:
        raise ValidationError(f"Mapping set has validation errors: {results}.")
    for res in results.results:
        r = ValidationResult("sssom_linkml_reference_validator")
        r.check = "TBD"
        r.data = res
        r.message = str(res)
        validation_results.add_result(r)
    return validation_results


def validate_shacl(msdf: MappingSetDataFrame) -> ValidationReport:
    """Validate SCHACL file.

    :param msdf: TODO: https://github.com/linkml/linkml/issues/850 .
    :raises NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError


def check_all_prefixes_in_curie_map(msdf: MappingSetDataFrame, fail_hard=True) -> ValidationReport:
    """Check all `EntityReference` slots are mentioned in 'curie_map'.

    :param msdf: MappingSetDataFrame
    :param fail_hard: If true, validation will fail on first error
    :raises ValidationError: If all prefixes not in curie_map.
    """
    prefixes = get_all_prefixes(msdf)
    prefixes_including_builtins = add_built_in_prefixes_to_prefix_map(msdf.prefix_map)
    added_built_in = {
        k: v for k, v in prefixes_including_builtins.items() if k not in msdf.prefix_map.keys()
    }
    if len(added_built_in) > 0:
        logging.info(f"Adding prefixes: {added_built_in} to the MapingSetDataFrame.")
    msdf.prefix_map = prefixes_including_builtins

    missing_prefixes = []
    for pref in prefixes:
        if pref != "" and pref not in list(msdf.prefix_map.keys()):
            missing_prefixes.append(pref)
    if missing_prefixes and fail_hard:
        raise ValidationError(f"The prefixes in {missing_prefixes} are missing from 'curie_map'.")
    report = ValidationReport("sssom_missing_prefixes")
    return report
