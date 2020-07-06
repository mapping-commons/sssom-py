"""
Converts sssom meta tsv to biolinkml
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Set, List, Union, Dict, Any
import pandas as pd
import logging

@dataclass
class MetaTSVConverter:
    """
    converts SSSOM/sssom_metadata.tsv
    """

    df: Optional[pd.DataFrame] = None

    def load(self, filename) -> None:
        """
        loads from folder
        :return:
        """
        self.df = pd.read_csv(filename, sep="\t", comment="#").fillna("")

    def convert(self) -> Dict[str, Any]:
        # note that 'mapping' is both a metaproperty and a property of this model...
        slots = {
            'mappings': {
                'description': 'Contains a list of mapping objects',
                'range': 'mapping',
                'multivalued': True
            }
        }
        classes = {
            'mapping set': {
                'description': 'Represents a set of mappings',
                'slots': ['mappings']
            },
            'mapping': {
                'description': 'Represents an individual mapping between a pair of entities',
                'slots': [],
                'class_uri': 'owl:Axiom'
            },
            'entity': {
                'description': 'Represents any entity that can be mapped, such as an OWL class or SKOS concept',
                'mappings': [
                    'rdf:Resource'
                ]
            }
        }
        obj = {
            'id': 'http://example.org/sssom',
            'description': 'Datamodel for Simple Standard for Sharing Ontology Mappings (SSSOM)',
            'imports': [
                'biolinkml:types'
            ],
            'prefixes': {
                'biolinkml': 'https://w3id.org/biolink/biolinkml/',
                'sssom': 'http://example.org/sssom/',

            },
            'see_also': ['https://github.com/OBOFoundry/SSSOM'],
            'default_curi_maps': ['semweb_context'],
            'default_prefix': 'sssom',
            'slots': slots,
            'classes': classes
        }
        for _, row in self.df.iterrows():
            id = row['Element ID']
            if id == 'ID':
                continue
            id = id.replace("sssom:", "")
            dt = row['Datatype']
            if dt == 'xsd:double':
                dt = 'double'
            elif id.endswith('_id') or id.endswith('match_field'):
                dt = 'entity'
            else:
                dt = 'string'

            slot = {
                'description': row['Description']
            }
            ep = row['Equivalent property']
            if ep != "":
                slot['mappings'] = [ep]
            if row['Required'] == 1:
                slot['required'] = True

            slot['range'] = dt
            slots[id] = slot
            slot_uri = None
            if id == 'subject_id':
                slot_uri = 'owl:annotatedSource'
            elif id == 'object_id':
                slot_uri = 'owl:annotatedTarget'
            elif id == 'predicate_id':
                slot_uri = 'owl:annotatedProperty'
            if slot_uri is not None:
                slot['slot_uri'] = slot_uri
            scope = row['Scope']
            if 'G' in scope:
                classes['mapping set']['slots'].append(id)
            if 'L' in scope:
                classes['mapping']['slots'].append(id)
        return obj

    def convert_and_save(self, fn: str) -> None:
        obj = self.convert()
        with open(fn, 'w') as stream:
            yaml.safe_dump(obj, stream, sort_keys=False)
