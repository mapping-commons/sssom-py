"""
Converts sssom meta tsv to linkml
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Set, List, Union, Dict, Any
import pandas as pd
from sssom.sssom_datamodel import Entity, slots
import logging
from io import StringIO
from .sssom_document import MappingSetDocument

@dataclass
class MappingSetDataFrame:
    """
    A collection of mappings represented as a DataFrame, together with additional metadata
    """

    df: pd.DataFrame = None ## Mappings
    prefixmap: Dict[str,str] = None ## maps CURIE prefixes to URI bases
    metadata: Optional[Dict[str,str]] = None ## header metadata excluding prefixes

@dataclass
class EntityPair:
    """
    A tuple of entities.

    Note that (e1,e2) == (e2,e1)
    """
    subject_entity: Entity
    object_entity: Entity

    def __hash__(self):
        if self.subject_entity.id <= self.object_entity.id:
            t = self.subject_entity.id, self.object_entity.id
        else:
            t = self.object_entity.id, self.subject_entity.id
        return hash(t)

@dataclass
class MappingSetDiff:
    """
    represents a difference between two mapping sets

    Currently this is limited to diffs at the level of entity-pairs.
    For example, if file1 has A owl:equivalentClass B, and file2 has A skos:closeMatch B,
    this is considered a mapping in common.
    """
    unique_tuples1: Optional[Set[str]] = None
    unique_tuples2: Optional[Set[str]] = None
    common_tuples: Optional[Set[str]] = None

    combined_dataframe: Optional[pd.DataFrame] = None
    """
    Dataframe that combines with left and right dataframes with information injected into
    the comment column
    """




@dataclass
class MetaTSVConverter:
    """
    converts SSSOM/sssom_metadata.tsv
    DO NOT USE, NOW DEPRECATED!
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
            },
            'id': {
                'description': 'CURIE or IRI identifier',
                'identifier': True
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
                ],
                'slots': ['id']
            }
        }
        obj = {
            'id': 'http://w3id.org/sssom',
            'description': 'Datamodel for Simple Standard for Sharing Ontology Mappings (SSSOM)',
            'imports': [
                'linkml:types'
            ],
            'prefixes': {
                'linkml': 'https://w3id.org/linkml/',
                'sssom': 'http://w3id.org/sssom/',

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

@dataclass
class MappingSetDataFrame:
    """
    A collection of mappings represented as a DataFrame, together with additional metadata
    """
    df: pd.DataFrame = None ## Mappings
    prefixmap: Dict[str,str] = None ## maps CURIE prefixes to URI bases
    metadata: Optional[Dict[str,str]] = None ## header metadata excluding prefixes

    
def get_file_extension(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        return f_format
    else:
        raise Exception(f'Cannot guess format from {filename}')

def read_csv(filename, comment='#', sep=','):
    lines = "".join([line for line in open(filename) 
                    if not line.startswith(comment)])
    return pd.read_csv(StringIO(lines), sep=sep)

def read_pandas(filename: str, sep=None) -> pd.DataFrame:
    """
    wrapper to pd.read_csv that handles comment lines correctly
    :param filename:
    :param sep: File separator in pandas (\t or ,)
    :return:
    """
    if not sep:
        extension = get_file_extension(filename)
        sep = "\t"
        if extension == "tsv":
            sep = "\t"
        elif extension == "csv":
            sep = ","
        else:
            logging.warning(f"Cannot automatically determine table format, trying tsv.")

    # from tempfile import NamedTemporaryFile
    # with NamedTemporaryFile("r+") as tmp:
    #    with open(filename, "r") as f:
    #        for line in f:
    #            if not line.startswith('#'):
    #                tmp.write(line + "\n")
    #    tmp.seek(0)
    return read_csv(filename, comment='#', sep=sep).fillna("")

def extract_global_metadata(msdoc: MappingSetDocument):
    meta = {'curie_map': msdoc.curie_map}
    ms_meta = msdoc.mapping_set
    for key in [slot for slot in dir(slots) if not callable(getattr(slots, slot)) and not slot.startswith("__")]:
        slot = getattr(slots, key).name
        if slot not in ["mappings"] and slot in ms_meta:
            if ms_meta[slot]:
                meta[key] = ms_meta[slot]
    return meta

def to_mapping_set_dataframe(doc:MappingSetDocument) -> MappingSetDataFrame:
    ###
    # convert MappingSetDocument into MappingSetDataFrame
    ###
    data = []
    for mapping in doc.mapping_set.mappings:
        mdict = mapping.__dict__
        m = {}
        for key in mdict:
            if mdict[key]:
                m[key] = mdict[key]
        data.append(m)
    df = pd.DataFrame(data=data)
    meta = extract_global_metadata(doc)
    msdf = MappingSetDataFrame(df=df, prefixmap=doc.curie_map, metadata=meta)
    return msdf

def to_mapping_set_document(doc:MappingSetDataFrame) -> MappingSetDocument:
    # Use standard method to do this.
    ###
    # convert MappingSetDataFrame into MappingSetDocument  
    '''df: pd.DataFrame = None ## Mappings
    prefixmap: Dict[str,str] = None ## maps CURIE prefixes to URI bases
    metadata: Optional[Dict[str,str]] = None ## header metadata excluding prefixes'''
    # TO
    '''mapping_set: MappingSet
    curie_map: Dict[str, str]'''
    # MappingSet =
    '''
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = SSSOM.MappingSet
    class_class_curie: ClassVar[str] = "sssom:MappingSet"
    class_name: ClassVar[str] = "mapping set"
    class_model_uri: ClassVar[URIRef] = SSSOM.MappingSet

    mappings: Optional[Union[Union[dict, "Mapping"], List[Union[dict, "Mapping"]]]] = empty_list()
    mapping_set_id: Optional[Union[str, EntityId]] = None
    mapping_set_version: Optional[str] = None
    creator_id: Optional[Union[str, EntityId]] = None
    creator_label: Optional[str] = None
    license: Optional[str] = None
    subject_source: Optional[str] = None
    subject_source_version: Optional[str] = None
    object_source: Optional[str] = None
    object_source_version: Optional[str] = None
    mapping_provider: Optional[str] = None
    mapping_tool: Optional[str] = None
    mapping_date: Optional[str] = None
    subject_match_field: Optional[Union[str, EntityId]] = None
    object_match_field: Optional[Union[str, EntityId]] = None
    subject_preprocessing: Optional[str] = None
    object_preprocessing: Optional[str] = None
    match_term_type: Optional[str] = None
    see_also: Optional[str] = None
    other: Optional[str] = None
    comment: Optional[str] = None
    '''
    ###
    df = MappingSetDocument(mapping_set=doc.df.to_dict(), curie_map=doc.metadata['curie_map'])
    return df
