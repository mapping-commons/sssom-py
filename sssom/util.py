import hashlib
import logging
from dataclasses import dataclass, field
from io import StringIO
import re
import sys
from typing import Any, Dict, List, Optional, Set, Union
import contextlib

import pandas as pd
import yaml
from scipy.stats.stats import normaltest
from sssom.sssom_datamodel import Entity, slots
from sssom.sssom_document import MappingSetDocument

from .sssom_document import MappingSetDocument

SSSOM_READ_FORMATS = ['tsv', 'rdf', 'owl', 'alignment-api-xml', 'obographs-json', 'json']
SSSOM_EXPORT_FORMATS = ['tsv', 'rdf', 'owl', 'json']


# TODO: use sssom_datamodel (Mapping Class)
SUBJECT_ID = 'subject_id'
SUBJECT_LABEL = 'subject_label'
OBJECT_ID = 'object_id'
OBJECT_LABEL = 'object_label'
PREDICATE_ID = 'predicate_id'
CONFIDENCE = 'confidence'
SUBJECT_CATEGORY = 'subject_category'
OBJECT_CATEGORY = 'object_category'
SUBJECT_SOURCE = 'subject_source'
OBJECT_SOURCE = 'object_source'
COMMENT = 'comment'
MAPPING_PROVIDER = 'mapping_provider'
MATCH_TYPE = 'match_type'
HUMAN_CURATED_MATCH_TYPE = 'HumanCurated'

_defining_features = [SUBJECT_ID, PREDICATE_ID, OBJECT_ID]

@dataclass
class MappingSetDataFrame:
    """
    A collection of mappings represented as a DataFrame, together with additional metadata
    """
    df: pd.DataFrame = None  ## Mappings
    prefixmap: Dict[str, str] = None  ## maps CURIE prefixes to URI bases
    metadata: Optional[Dict[str, str]] = None  ## header metadata excluding prefixes

    def merge(self, msdf2):
        """Merges two MappingSetDataframes

        Args:
            msdf2 (MappingSetDataFrame): Secondary MappingSetDataFrame (self => primary)

        Returns:
            MappingSetDataFrame: Merged MappingSetDataFrame
        """
        msdf = merge_msdf(msdf1=self, msdf2=msdf2)
        self.df = msdf.df
        self.prefixmap = msdf.prefixmap
        self.metadata = msdf.metadata

    def clean_prefix_map(self):
        prefixes_in_map = get_prefixes_used_in_table(self.df)
        new_prefixes = dict()
        missing_prefix = False
        for prefix in prefixes_in_map:
            if prefix in self.prefixmap:
                new_prefixes[prefix]=self.prefixmap[prefix]
            else:
                logging.warning(f"{prefix} is used in the data frame but does not exist in prefix map")
                missing_prefix = True
        if not missing_prefix:
            self.prefixmap = new_prefixes

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
        #self.df = pd.read_csv(filename, sep="\t", comment="#").fillna("")
        self.df = read_pandas(filename)


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


def parse(filename) -> pd.DataFrame:
    """
    parses a TSV to a pandas frame
    """
    # return from_tsv(filename)
    logging.info(f'Parsing {filename}')
    return pd.read_csv(filename, sep="\t", comment="#")
    # return read_pandas(filename)


def collapse(df):
    """
    collapses rows with same S/P/O and combines confidence
    """
    df2 = df.groupby([SUBJECT_ID, PREDICATE_ID, OBJECT_ID])[CONFIDENCE].apply(max).reset_index()
    return df2

def sort_sssom_columns(columns: list)-> list:
    # Ideally, the order of the sssom column names is parsed strictly from sssom.yaml

    logging.warning('SSSOM sort columns not implemented')
    columns.sort()
    return columns

def sort_sssom(df:pd.DataFrame)->pd.DataFrame:
    df.sort_values(by=sort_sssom_columns(list(df.columns)), ascending = False, inplace=True)
    return df

def filter_redundant_rows(df: pd.DataFrame, ignore_predicate=False) -> pd.DataFrame:
    """
    removes rows if there is another row with same S/O and higher confidence

    :param df:
    :return:
    """
    # tie-breaker
    # create a 'sort' method and then replce the following line by sort()
    df = sort_sssom(df)
    #df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    if ignore_predicate:
        key = [SUBJECT_ID, OBJECT_ID]
    else:
        key = [SUBJECT_ID, OBJECT_ID, PREDICATE_ID]
    dfmax:pd.DataFrame
    dfmax = df.groupby(key, as_index=False)[CONFIDENCE].apply(max).drop_duplicates()
    max_conf = {}
    for index, row in dfmax.iterrows():
        if ignore_predicate:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID])] = row[CONFIDENCE]
        else:
            max_conf[(row[SUBJECT_ID], row[OBJECT_ID], row[PREDICATE_ID])] = row[CONFIDENCE]
    if ignore_predicate:
        return df[df.apply(lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID])], axis=1)]
    else:
        return df[df.apply(lambda x: x[CONFIDENCE] >= max_conf[(x[SUBJECT_ID], x[OBJECT_ID], x[PREDICATE_ID])], axis=1)]


def remove_unmatched(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where no match is found. TODO: https://github.com/OBOFoundry/SSSOM/issues/28
    :param df:
    :return:
    """
    return df[df[PREDICATE_ID] != 'noMatch']


def create_entity(row, id: str, mappings: Dict) -> Entity:
    e = Entity(id=id)
    for k, v in mappings.items():
        if k in e:
            e[k] = v
    return e


def group_mappings(df: pd.DataFrame) -> Dict[EntityPair, List]:
    """
    group mappings by EntityPairs
    """
    mappings: Dict = {}
    for _, row in df.iterrows():
        sid = row[SUBJECT_ID]
        oid = row[OBJECT_ID]
        s = create_entity(row, sid, {
            'label': SUBJECT_LABEL,
            'category': SUBJECT_CATEGORY,
            'source': SUBJECT_SOURCE
        })
        o = create_entity(row, oid, {
            'label': OBJECT_LABEL,
            'category': OBJECT_CATEGORY,
            'source': OBJECT_SOURCE
        })
        pair = EntityPair(s, o)
        if pair not in mappings:
            mappings[pair] = []
        mappings[pair].append(row)
    return mappings


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> MappingSetDiff:
    """
    Perform a diff between two SSSOM dataframes

    Currently does not discriminate between mappings with different predicates
    """
    mappings1 = group_mappings(df1.copy())
    mappings2 = group_mappings(df2.copy())
    tuples1 = set(mappings1.keys())
    tuples2 = set(mappings2.keys())
    d = MappingSetDiff()
    d.unique_tuples1 = tuples1.difference(tuples2)
    d.unique_tuples2 = tuples2.difference(tuples1)
    d.common_tuples = tuples1.intersection(tuples2)
    all_tuples = tuples1.union(tuples2)
    all_ids = set()
    for t in all_tuples:
        all_ids.update({t.subject_entity.id, t.object_entity.id})
    rows = []
    for t in d.unique_tuples1:
        for r in mappings1[t]:
            r[COMMENT] = 'UNIQUE_1'
        rows += mappings1[t]
    for t in d.unique_tuples2:
        for r in mappings2[t]:
            r[COMMENT] = 'UNIQUE_2'
        rows += mappings2[t]
    for t in d.common_tuples:
        new_rows = mappings1[t] + mappings2[t]
        for r in new_rows:
            r[COMMENT] = 'COMMON_TO_BOTH'
        rows += new_rows
    # for r in rows:
    #    r['other'] = 'synthesized sssom file'
    d.combined_dataframe = pd.DataFrame(rows)
    return d


@contextlib.contextmanager
def smart_open(filename=None):
    # https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
    if filename and filename != '-':
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()


def dataframe_to_ptable(df: pd.DataFrame, priors=[0.02, 0.02, 0.02, 0.02], inverse_factor: float = 0.5):
    """
    exports kboom ptable
    :param df: SSSOM dataframe
    :param inverse_factor: relative weighting of probability of inverse of predicate
    :return:
    """
    df = collapse(df)
    rows = []
    for _, row in df.iterrows():
        s = row[SUBJECT_ID]
        o = row[OBJECT_ID]
        c = row[CONFIDENCE]
        # confidence of inverse
        # e.g. if Pr(super) = 0.2, then Pr(sub) = (1-0.2) * IF
        ic = (1.0 - c) * inverse_factor
        # residual confidence
        rc = (1 - (c + ic)) / 2.0

        p = row[PREDICATE_ID]
        if p == 'owl:equivalentClass':
            pi = 2
        elif p == 'skos:exactMatch':
            pi = 2
        elif p == 'skos:closeMatch':
            # TODO: consider distributing
            pi = 2
        elif p == 'owl:subClassOf':
            pi = 0
        elif p == 'skos:broadMatch':
            pi = 0
        elif p == 'inverseOf(owl:subClassOf)':
            pi = 1
        elif p == 'skos:narrowMatch':
            pi = 1
        elif p == 'owl:differentFrom':
            pi = 3
        elif p == 'dbpedia-owl:different':
            pi = 3
        else:
            #raise Exception(f'Unknown predicate {p}')
            logging.warning(f'Unknown predicate {p}')


        if pi == 0:
            # subClassOf
            ps = (c, ic, rc, rc)
        elif pi == 1:
            # superClassOf
            ps = (ic, c, rc, rc)
        elif pi == 2:
            # equivalent
            ps = (rc, rc, c, ic)
        elif pi == 3:
            # sibling
            ps = (rc, rc, ic, c)
        else:
            raise Exception(f'pi: {pi}')
        row = [s, o] + [str(p) for p in ps]
        rows.append(row)
    return rows


RDF_FORMATS = ['ttl', 'turtle', 'nt']


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

def merge_msdf(msdf1:MappingSetDataFrame, msdf2:MappingSetDataFrame, reconcile:bool=True, inplace:bool=False) -> MappingSetDataFrame:
        """
        Merging msdf2 into msdf1,
        if reconcile=True, then dedupe(remove redundant lower confidence mappings) and
            reconcile (if msdf contains a higher confidence _negative_ mapping,
            then remove lower confidence positive one. If confidence is the same,
            prefer HumanCurated. If both HumanCurated, prefer negative mapping).

        Args:
            msdf1 (MappingSetDataFrame): The primary MappingSetDataFrame
            msdf2 (MappingSetDataFrame): The secondary MappingSetDataFrame
            reconcile (bool, optional): [description]. Defaults to True.

        Returns:
            MappingSetDataFrame: Merged MappingSetDataFrame.
        """
        # Inject metadata of msdf into df
        msdf1 = inject_metadata_into_df(msdf=msdf1)
        msdf2 = inject_metadata_into_df(msdf=msdf2)

        merged_msdf = MappingSetDataFrame()
        # If msdf2 has a DataFrame
        if msdf2.df is not None:
            # 'outer' join in pandas == FULL JOIN in SQL
            merged_msdf.df = msdf1.df.merge(msdf2.df, how='outer')
        else:
            merged_msdf.df = msdf1.df
        #merge the non DataFrame elements
        merged_msdf.prefixmap = dict_merge(msdf2.prefixmap, msdf1.prefixmap, 'prefixmap')
        # After a Slack convo with @matentzn, commented out below.
        #merged_msdf.metadata = dict_merge(msdf2.metadata, msdf1.metadata, 'metadata')


        '''if inplace:
            msdf1.prefixmap = merged_msdf.prefixmap
            msdf1.metadata = merged_msdf.metadata
            msdf1.df = merged_msdf.df'''

        if reconcile:
            merged_msdf.df = filter_redundant_rows(merged_msdf.df)

            merged_msdf.df = deal_with_negation(merged_msdf.df) #deals with negation

        return merged_msdf

def deal_with_negation(df:pd.DataFrame)-> pd.DataFrame:
        """Combine negative and positive rows with matching [SUBJECT_ID, OBJECT_ID, CONFIDENCE] combination
        taking into account the rule that negative trumps positive given equal confidence values.

        Args:
            df (pd.DataFrame): Merged Pandas DataFrame

        Returns:
            pd.DataFrame: Pandas DataFrame with negations addressed
        """

        '''
            1. Mappings in mapping1 trump mappings in mapping2 (if mapping2 contains a conflicting mapping in mapping1,
               the one in mapping1 is preserved).
            2. Reconciling means two things
                [i] if the same s,p,o (subject_id, object_id, predicate_id) is present multiple times,
                    only preserve the highest confidence one. If confidence is same, rule 1 (above) applies.
                [ii] If s,!p,o and s,p,o , then prefer higher confidence and remove the other.
                     If same confidence prefer "HumanCurated" .If same again prefer negative.
            3. Prefixes:
                [i] if there is the same prefix in mapping1 as in mapping2, and the prefix URL is different, throw an error and fail hard
                    else just merge the two prefix maps
            4. Metadata: same as rule 1.

            #1; #2(i) #3 and $4 are taken care of by 'filtered_merged_df' Only #2(ii) should be performed here.
        '''

        ######  If s,!p,o and s,p,o , then prefer higher confidence and remove the other.  ###
        negation_df:pd.DataFrame
        negation_df = df.loc[df[PREDICATE_ID].str.startswith('!')]# or df.loc[df['predicate_modifier'] == 'NOT']

        # #####This step ONLY if 'NOT' is expressed by the symbol '!' in 'predicate_id' #####
        normalized_negation_df = negation_df.reset_index()
        normalized_negation_df[PREDICATE_ID] = normalized_negation_df[PREDICATE_ID].str.replace('!','')
        ########################################################
        normalized_negation_df = normalized_negation_df.drop(['index'], axis = 1)

        # remove the NOT rows from the main DataFrame
        condition = negation_df.isin(df)
        positive_df = df.drop(condition.index)
        positive_df = positive_df.reset_index().drop(['index'], axis=1)

        columns_of_interest = [SUBJECT_ID,PREDICATE_ID, OBJECT_ID, CONFIDENCE, MATCH_TYPE]
        negation_subset = normalized_negation_df[columns_of_interest]
        positive_subset = positive_df[columns_of_interest]

        combined_normalized_subset = pd.concat([positive_subset, negation_subset]).drop_duplicates()

        #GroupBy and SELECT ONLY maximum confidence
        max_confidence_df:pd.DataFrame
        max_confidence_df = combined_normalized_subset.groupby(_defining_features, as_index=False)[CONFIDENCE].max()

        ####### If same confidence prefer "HumanCurated". ################
        reconciled_df_subset:pd.DataFrame
        reconciled_df_subset = pd.DataFrame(columns=combined_normalized_subset.columns)
        for idx_1, row_1 in max_confidence_df.iterrows():
            match_condition_1 = (combined_normalized_subset[SUBJECT_ID] == row_1[SUBJECT_ID]) & \
                              (combined_normalized_subset[OBJECT_ID] == row_1[OBJECT_ID]) & \
                              (combined_normalized_subset[CONFIDENCE] == row_1[CONFIDENCE])
            # match_condition_1[match_condition_1] gives the list of 'True's.
            # In other words, the rows that match the condition (rules declared).
            # Ideally, there should be 1 row. If not apply an extra rule to look for 'HumanCurated'.
            if len(match_condition_1[match_condition_1].index) > 1:
                match_condition_1 = (combined_normalized_subset[SUBJECT_ID] == row_1[SUBJECT_ID]) & \
                                  (combined_normalized_subset[OBJECT_ID] == row_1[OBJECT_ID]) & \
                                  (combined_normalized_subset[CONFIDENCE] == row_1[CONFIDENCE]) & \
                                  (combined_normalized_subset[MATCH_TYPE] == HUMAN_CURATED_MATCH_TYPE)
            # TODO: In spite of this returning multiple rows, pick the 1st row.


            reconciled_df_subset = reconciled_df_subset.append(combined_normalized_subset.loc[match_condition_1[match_condition_1].index, :])



        # Add negations (NOT symbol) back to the PREDICATE_ID
        # NOTE: negative TRUMPS positive if negative and positive with same
        # [SUBJECT_ID, OBJECT_ID, PREDICATE_ID] exist
        for idx_2, row_2 in negation_df.iterrows():
            match_condition_2 = (reconciled_df_subset[SUBJECT_ID] == row_2[SUBJECT_ID]) & \
                                (reconciled_df_subset[OBJECT_ID] == row_2[OBJECT_ID]) & \
                                (reconciled_df_subset[CONFIDENCE] == row_2[CONFIDENCE])
            reconciled_df_subset.loc[match_condition_2[match_condition_2].index, PREDICATE_ID] = row_2[PREDICATE_ID]

        reconciled_df:pd.DataFrame
        reconciled_df = pd.DataFrame(columns=df.columns)
        for idx_3, row_3 in reconciled_df_subset.iterrows():
            match_condition_3 = (df[SUBJECT_ID] == row_3[SUBJECT_ID]) & \
                                (df[OBJECT_ID] == row_3[OBJECT_ID]) & \
                                (df[CONFIDENCE] == row_3[CONFIDENCE]) & \
                                (df[PREDICATE_ID] == row_3[PREDICATE_ID])
            reconciled_df = reconciled_df.append(df.loc[match_condition_3[match_condition_3].index, :])

        return reconciled_df

def dict_merge(source:Dict, target:Dict, dict_name:str) -> Dict:
    """
    Takes 2 MappingSetDataFrame elements (prefixmap OR metadata) and merges source => target

    Args:
        source (Dict): MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
        target (Dict): MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
        dict_name (str): prefixmap or metadata

    Returns:
        Dict: merged MappingSetDataFrame.prefixmap / MappingSetDataFrame.metadata
    """
    if source is not None:
        k:str
        for k, v in source.items():
                if k not in target:
                    if v not in list(target.values()):
                        target[k] = v
                    else:
                        common_values = [i for i,val in target.items() if val == v]
                        raise ValueError(f'Value [{v}] is present in {dict_name} for multiple keys [{common_values}].')
                else:
                    if target[k] != v:
                        raise ValueError(f'{dict_name} values in both MappingSetDataFrames for the same key [{k}] are different.')
    return target

def inject_metadata_into_df(msdf:MappingSetDataFrame)->MappingSetDataFrame:
    """Inject metadata dictionary key-value pair into DataFrame columns in a MappingSetDataFrame.DataFrame.

    Args:
        msdf (MappingSetDataFrame): MappingSetDataFrame with metadata separate.

    Returns:
        MappingSetDataFrame: MappingSetDataFrame with metadata as columns
    """

    for k,v in msdf.metadata.items():
        if k not in msdf.df.columns:
            msdf.df[k] = v
    return msdf

def get_file_extension(filename: str) -> str:
    parts = filename.split(".")
    if len(parts) > 0:
        f_format = parts[-1]
        return f_format
    else:
        raise Exception(f'Cannot guess format from {filename}')


def read_csv(filename, comment='#', sep=','):
    with open(filename, 'r') as f:
        lines = "".join([line for line in f
                        if not line.startswith(comment)])
    return pd.read_csv(StringIO(lines), sep=sep)


def read_metadata(filename):
    """
    Reading metadata file (yaml) that is supplied separately from a tsv
    :param filename: location of file
    :return: two objects, a metadata and a curie_map object
    """
    meta = {}
    curie_map = {}
    with open(filename, 'r') as stream:
        try:
            m = yaml.safe_load(stream)
            if "curie_map" in m:
                curie_map = m['curie_map']
            m.pop('curie_map', None)
            meta = m
        except yaml.YAMLError as exc:
            print(exc)
    return meta, curie_map


def read_pandas(filename: str, sep='\t') -> pd.DataFrame:
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


def to_mapping_set_dataframe(doc: MappingSetDocument) -> MappingSetDataFrame:
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
    meta.pop("curie_map", None)
    msdf = MappingSetDataFrame(df=df, prefixmap=doc.curie_map, metadata=meta)
    return msdf

# to_mapping_set_document is in parser.py in order to avoid circular import errors

def is_curie(string: str):
    return re.match(r"[A-Za-z0-9_]+[:][A-Za-z0-9_]", string)


def get_prefix_from_curie(curie: str):
    if is_curie(curie):
        return curie.split(":")[0]
    else:
        return ''


def get_prefixes_used_in_table(df: pd.DataFrame):
    prefixes = []
    for col in _defining_features:
        for v in df[col].values:
            prefixes.append(get_prefix_from_curie(v))
    return list(set(prefixes))