import hashlib
import logging
import random
from typing import Dict, List

import pandas as pd
from scipy.stats.stats import normaltest

from sssom.datamodel_util import MappingSetDataFrame, MappingSetDiff, EntityPair
from sssom.sssom_datamodel import Entity

# TODO: use sssom_datamodel
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
    df2 = df.groupby(_defining_features)[CONFIDENCE].apply(max).reset_index()
    return df2


def filter_redundant_rows(df: pd.DataFrame, ignore_predicate=False) -> pd.DataFrame:
    """
    removes rows if there is another row with same S/O and higher confidence

    :param df:
    :return:
    """
    # tie-breaker
    df[CONFIDENCE] = df[CONFIDENCE].apply(lambda x: x + random.random() / 10000)
    if ignore_predicate:
        key = [SUBJECT_ID, OBJECT_ID]
    else:
        key = [SUBJECT_ID, OBJECT_ID, PREDICATE_ID]
    dfmax = df.groupby(key)[CONFIDENCE].apply(max).reset_index()
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
            print(f'Unknown predicate {p}')


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
        

        merged_msdf = MappingSetDataFrame()
        # If msdf2 has a DataFrame
        if msdf2.df is not None:
            # 'outer' join in pandas == FULL JOIN in SQL
            merged_msdf.df = msdf1.df.merge(msdf2.df, how='outer', on=_defining_features)
        else:
            merged_msdf.df = msdf1.df
        #merge the non DataFrame elements
        merged_msdf.prefixmap = dict_merge(msdf2.prefixmap, msdf1.prefixmap, 'prefixmap')
        merged_msdf.metadata = dict_merge(msdf2.metadata, msdf1.metadata, 'prefixmap')


        '''if inplace:
            msdf1.prefixmap = merged_msdf.prefixmap
            msdf1.metadata = merged_msdf.metadata
            msdf1.df = merged_msdf.df'''

        if reconcile:
            merged_msdf.df = filter_redundant_rows(merged_msdf.df)
            
            merged_msdf.df = deal_with_negation(merged_msdf.df) #deals with negation
                        
        return merged_msdf

def deal_with_negation(df:pd.DataFrame)-> pd.DataFrame:
        """[summary]

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
            if len(match_condition_1[match_condition_1].index) > 1:
                match_condition_1 = (combined_normalized_subset[SUBJECT_ID] == row_1[SUBJECT_ID]) & \
                                  (combined_normalized_subset[OBJECT_ID] == row_1[OBJECT_ID]) & \
                                  (combined_normalized_subset[CONFIDENCE] == row_1[CONFIDENCE]) & \
                                  (combined_normalized_subset[MATCH_TYPE] == HUMAN_CURATED_MATCH_TYPE)

            reconciled_df_subset = reconciled_df_subset.append(combined_normalized_subset.loc[match_condition_1[match_condition_1].index, :])

                    
                
        # Add negations (NOT symbol) back to the PREDICATE_ID
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
        for k, v in source:
                if k not in target:
                    target[k] = v
                else:
                    if target[k] != v:
                        raise ValueError(f'.{dict_name} values in both MappingSetDataFrames for the same key [{k}] are different.')
    return target