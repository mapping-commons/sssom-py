import logging

from jsonschema import ValidationError
from linkml.validators.jsonschemavalidator import JsonSchemaDataValidator
from sssom_schema import MappingSet

from sssom.parsers import to_mapping_set_document
from sssom.util import MappingSetDataFrame, get_prefix_from_curie

from .constants import ENTITY_REFERENCE_SLOTS, SCHEMA_VIEW, SCHEMA_YAML


def json_schema_validate(msdf: MappingSetDataFrame) -> bool:
    """_summary_

    :param msdf: MappingSetDataFrame.
    :return: Validation result (True/False).
    """
    try:
        validator = JsonSchemaDataValidator(SCHEMA_YAML)
        mapping_set = to_mapping_set_document(msdf).mapping_set
        validator.validate_object(mapping_set, MappingSet)
        check_all_prefixes_in_curie_map(msdf)
        return True
    except Exception as e:
        logging.exception(e)
        return False


def check_all_prefixes_in_curie_map(msdf: MappingSetDataFrame) -> None:
    """Check all `EntityReference` slots are mentioned in 'curie_map'.

    :param msdf: MappingSetDataFrame
    :raises ValidationError: If all prefixes not in curie_map.
    :return: None
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
    :return: List of all prefixes.
    """
    metadata_keys = list(msdf.metadata.keys())
    df_columns_list = msdf.df.columns.to_list()
    all_keys = metadata_keys + df_columns_list
    ent_ref_slots = [s for s in all_keys if s in ENTITY_REFERENCE_SLOTS]
    prefix_list = []
    for slot in ent_ref_slots:
        if slot in metadata_keys:
            if type(msdf.metadata[slot]) == list:
                for s in msdf.metadata[slot]:
                    if get_prefix_from_curie(s) == "":
                        print(
                            f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}"
                        )
                        # raise ValidationError(f"Slot '{slot}' has an incorrect value: {msdf.metadata[s]}")
                    prefix_list.append(get_prefix_from_curie(s))
            else:
                if get_prefix_from_curie(msdf.metadata[slot]) == "":
                    print(
                        f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}"
                    )
                    # raise ValidationError(f"Slot '{slot}' has an incorrect value: {msdf.metadata[slot]}")
                prefix_list.append(get_prefix_from_curie(msdf.metadata[slot]))
        else:
            column_prefixes = list(
                set(
                    [
                        get_prefix_from_curie(s)
                        for s in list(set(msdf.df[slot].to_list()))
                    ]
                )
            )
            prefix_list = prefix_list + column_prefixes

    prefix_list = list(set(prefix_list))

    return prefix_list
