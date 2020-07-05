from .sssom_datamodel import MappingSet, Mapping

from dataclasses import dataclass
from typing import Dict

@dataclass()
class MappingSetDocument():

    mapping_set: MappingSet

    curie_map: Dict[str, str]