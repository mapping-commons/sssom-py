"""Validators."""

import logging
from typing import List

from jsonschema import ValidationError
from linkml.validators.jsonschemavalidator import JsonSchemaDataValidator
from linkml.validators.sparqlvalidator import SparqlDataValidator  # noqa: F401
from sssom_schema import MappingSet

from sssom.constants import SchemaValidationType, SSSOMSchemaView
from sssom.context import add_built_in_prefixes_to_prefix_map
from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_all_prefixes


def validate(
    msdf: MappingSetDataFrame, validation_types: List[SchemaValidationType]
) -> None:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    """
    validation_methods = {
        SchemaValidationType.JsonSchema: validate_json_schema,
        SchemaValidationType.Shacl: validate_shacl,
        SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    }
    for vt in validation_types:
        validation_methods[vt](msdf)


def validate_json_schema(msdf: MappingSetDataFrame) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    """
    schema_view_object = SSSOMSchemaView()
    validator = JsonSchemaDataValidator(schema_view_object.yaml)
    mapping_set = to_mapping_set_document(msdf).mapping_set
    validator.validate_object(mapping_set, MappingSet)


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
    prefixes = get_all_prefixes(msdf)
    prefixes_including_builtins = add_built_in_prefixes_to_prefix_map(msdf.prefix_map)
    added_built_in = {
        k: v
        for k, v in prefixes_including_builtins.items()
        if k not in msdf.prefix_map.keys()
    }
    if len(added_built_in) > 0:
        logging.info(f"Adding prefixes: {added_built_in} to the MapingSetDataFrame.")
    msdf.prefix_map = prefixes_including_builtins

    missing_prefixes = []
    for pref in prefixes:
        if pref not in list(msdf.prefix_map.keys()):
            missing_prefixes.append(pref)
    if missing_prefixes:
        raise ValidationError(
            f"The prefixes in {missing_prefixes} are missing from 'curie_map'."
        )
