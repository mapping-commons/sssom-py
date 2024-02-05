"""Validators."""

from typing import Callable, List, Mapping

from jsonschema import ValidationError
from linkml_runtime.processing.referencevalidator import ReferenceValidator
from linkml_runtime.utils.schemaview import SchemaView
from sssom_schema import MappingSet

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


def validate_json_schema(msdf: MappingSetDataFrame) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    """
    validator = ReferenceValidator(SchemaView(SCHEMA_YAML))
    mapping_set = to_mapping_set_document(msdf).mapping_set
    validator.validate(mapping_set, MappingSet)


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
