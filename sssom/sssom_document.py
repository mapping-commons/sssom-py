from .sssom_datamodel import MappingSet, Mapping, Entity

from dataclasses import dataclass
from typing import Dict

@dataclass()
class MappingSetDocument():
    """
    Represents a single SSSOM document.

    A document is simply a holder for a MappingSet object plus a CURIE map
    """

    mapping_set: MappingSet
    """
    The main part of the document: a set of mappings plus metadata
    """

    curie_map: Dict[str, str]
    """
    Mappings between ID prefixes and URI Bases, used to map CURIEs to URIs.
    Note that the CURIE map is not part of the core SSSOM model, hence it belongs here in the document
    object
    """