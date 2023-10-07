# -*- coding: utf-8 -*-

"""Type hints for SSSOM."""

from collections import ChainMap
from pathlib import Path
from typing import Optional, Tuple, Union

import curies
import yaml
from curies import Converter

from .constants import (
    CURIE_MAP,
    PREFIX_MAP_MODE_MERGED,
    PREFIX_MAP_MODE_METADATA_ONLY,
    PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY,
    MetadataType,
    get_default_metadata,
)
from .context import get_converter


def _parse_file_metadata_helper(
    metadata_path: Union[None, str, Path] = None, prefix_map_mode: Optional[str] = None
) -> Tuple[Converter, MetadataType]:
    """
    Load SSSOM metadata from a file, and then augments it with default prefixes.

    :param metadata_path: The metadata file in YAML format
    :param prefix_map_mode: one of metadata_only, sssom_default_only, merged
    :return: a prefix map dictionary and a metadata object dictionary
    """
    if metadata_path is None:
        return get_converter(), get_default_metadata()

    with Path(metadata_path).resolve().open() as file:
        metadata = yaml.safe_load(file)

    metadata = dict(ChainMap(metadata, get_default_metadata()))

    converter = Converter.from_prefix_map(metadata.pop(CURIE_MAP, {}))

    # FIXME just remove this functionality and use the same chain as everywhere else
    if prefix_map_mode is None or prefix_map_mode == PREFIX_MAP_MODE_METADATA_ONLY:
        pass
    elif prefix_map_mode == PREFIX_MAP_MODE_SSSOM_DEFAULT_ONLY:
        converter = get_converter()
    elif prefix_map_mode == PREFIX_MAP_MODE_MERGED:
        converter = curies.chain([converter, get_converter()])
    else:
        raise ValueError(f"Invalid prefix map mode: {prefix_map_mode}")
    return converter, metadata
