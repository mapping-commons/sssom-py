"""Validators."""
import logging
from typing import Callable, List, Mapping

from dataclasses import asdict
from jsonschema import ValidationError
from linkml.validator import Validator, ValidationReport
from linkml.validator.report import Severity
from linkml.validator.plugins import JsonschemaValidationPlugin, RecommendedSlotsPlugin

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
    validation_errors = 0

    if not report.results:
        logging.info('The instance is valid!')
    else:
        for result in report.results:
            validation_errors += 1
            if result.severity == Severity.ERROR:
                logging.error(result.message)

    if fail_on_error and validation_errors:
        raise ValidationError(f"You mapping set has {validation_errors} validation errors!")

def validate_json_schema(msdf: MappingSetDataFrame) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    """
    validator = Validator(
        schema=SCHEMA_YAML,
        validation_plugins=[
            JsonschemaValidationPlugin(closed=True),
            RecommendedSlotsPlugin()
        ]
    )
    mapping_set = to_mapping_set_document(msdf).mapping_set

    mapping_set_yaml = asdict(mapping_set)
    report = validator.validate(mapping_set_yaml, "mapping set")
    print_linkml_report(report)


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
