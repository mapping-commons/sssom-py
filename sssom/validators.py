"""Validators."""

from typing import List
from jsonschema import ValidationError
from linkml.validators.jsonschemavalidator import JsonSchemaDataValidator
from linkml.validators.sparqlvalidator import SparqlDataValidator  # noqa: F401
from sssom_schema import MappingSet

from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_prefix_from_curie

from .constants import ENTITY_REFERENCE_SLOTS, SCHEMA_YAML, SchemaValidationType


def validate(
    msdf: MappingSetDataFrame, validation_types: List[SchemaValidationType]
) -> None:
    """Validate SSSOM files against `sssom-schema` using linkML's validator function.

    :param msdf: MappingSetDataFrame.
    :param validation_types: SchemaValidationType
    :return: Validation error or None.
    """
    validation_methods = {
        SchemaValidationType.JsonSchema: validate_json_schema,
        SchemaValidationType.Shacl: validate_shacl,
        SchemaValidationType.PrefixMapCompleteness: check_all_prefixes_in_curie_map,
    }
    for vt in validation_types:
        return validation_methods[vt](msdf)


def validate_json_schema(msdf: MappingSetDataFrame) -> None:
    """Validate JSON Schema using linkml's JsonSchemaDataValidator.

    :param msdf: MappingSetDataFrame to eb validated.
    """
    validator = JsonSchemaDataValidator(SCHEMA_YAML)
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
    missing_prefixes = []
    for pref in prefixes:
        if pref not in list(msdf.prefix_map.keys()):
            missing_prefixes.append(pref)
    if missing_prefixes:
        raise ValidationError(
            f"The prefixes in {missing_prefixes} are missing from 'curie_map'."
        )


def get_all_prefixes(msdf: MappingSetDataFrame) -> list:
    """Fetch all prefixes in the MappingSetDataFrame.

    :param msdf: MappingSetDataFrame
    :raises ValidationError: If slot is wrong.
    :raises ValidationError: If slot is wrong.
    :return:  List of all prefixes.
    """
    prefix_list = []
    if msdf.metadata and not msdf.df.empty:  # type: ignore
        metadata_keys = list(msdf.metadata.keys())
        df_columns_list = msdf.df.columns.to_list()  # type: ignore
        all_keys = metadata_keys + df_columns_list
        ent_ref_slots = [s for s in all_keys if s in ENTITY_REFERENCE_SLOTS]

        for slot in ent_ref_slots:
            if slot in metadata_keys:
                if type(msdf.metadata[slot]) == list:
                    for s in msdf.metadata[slot]:
                        if get_prefix_from_curie(s) == "":
                            # print(
                            #     f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}"
                            # )
                            raise ValidationError(
                                f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}"
                            )
                        prefix_list.append(get_prefix_from_curie(s))
                else:
                    if get_prefix_from_curie(msdf.metadata[slot]) == "":
                        # print(
                        #     f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                        # )
                        raise ValidationError(
                            f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                        )
                    prefix_list.append(get_prefix_from_curie(msdf.metadata[slot]))
            else:
                column_prefixes = list(
                    set(
                        [
                            get_prefix_from_curie(s)
                            for s in list(set(msdf.df[slot].to_list()))  # type: ignore
                        ]
                    )
                )
                prefix_list = prefix_list + column_prefixes

        prefix_list = list(set(prefix_list))

    return prefix_list
