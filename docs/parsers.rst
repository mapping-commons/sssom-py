Parsers
========

Field descriptions
------------------

Taken from:

https://www.nlm.nih.gov/healthit/snomedct/us_edition.html

Download a zip file there, and inside there will be the following PDF, which documents the fields as shown below.

doc_Icd10cmMapReleaseNotes_Current-en-US_US1000124_20210901.pdf

More info:

https://www.nlm.nih.gov/research/umls/mapping_projects/snomedct_to_icd10cm.html

FIELD,DATA_TYPE,PURPOSE,SSSOM Dev Comments
- id,UUID,A 128 bit unsigned integer, uniquely identifying the map record,
- effectiveTime,Time,Specifies the inclusive date at which this change becomes effective.,
- active,Boolean,Specifies whether the member’s state was active (=1) or inactive (=0) from the nominal release date specified by the effectiveTime field.,
- moduleId,SctId,Identifies the member version’s module. Set to a child of 900000000000443000|Module| within the metadata hierarchy.,The only value in the entire set is '5991000124107', which has label 'SNOMED CT to ICD-10-CM rule-based mapping module' (https://www.findacode.com/snomed/5991000124107--snomed-ct-to-icd-10-cm-rule-based-mapping-module.html).
- refSetId,SctId,Set to one of the children of the |Complex map type| concept in the metadata hierarchy.,The only value in the entire set is '5991000124107', which has label 'ICD-10-CM complex map reference set' (https://www.findacode.com/snomed/6011000124106--icd-10-cm-complex-map-reference-set.html).
- referencedComponentId,SctId,The SNOMED CT source concept ID that is the subject of the map record.,
- mapGroup,Integer,An integer identifying a grouping of complex map records which will designate one map target at the time of map rule evaluation. Source concepts that require two map targets for classification will have two sets of map groups.,
- mapPriority,Integer,Within a map group, the mapPriority specifies the order in which complex map records should be evaluated to determine the correct map target.,
- mapRule,String,A machine-readable rule, (evaluating to either ‘true’ or ‘false’ at run-time) that indicates whether this map record should be selected within its map group.,
- mapAdvice,String,Human-readable advice that may be employed by the software vendor to give an end-user advice on selection of the appropriate target code. This includes a) a summary statement of the map rule logic, b) a statement of any limitations of the map record and c) additional classification guidance for the coding professional.,
- mapTarget,String,The target ICD-10 classification code of the map record.,
- correlationId,SctId,A child of |Map correlation value| in the metadata hierarchy, identifying the correlation between the SNOMED CT concept and the target code.,
- mapCategoryId,SctId,Identifies the SNOMED CT concept in the metadata hierarchy which is the MapCategory for the associated map record. This is a subtype of 447634004 |ICD-10 Map Category value|.,

Mappings: SSSOM::SNOMED_Complex_Map
-----------------------------------
Copy/pasta of state of mappings as of 2022/03/04:

'subject_id': f'SNOMED:{row["referencedComponentId"]}',
'subject_label': row['referencedComponentName'],

# 'predicate_id': 'skos:exactMatch',
# - mapCategoryId: can use for mapping predicate? Or is correlationId more suitable?
#   or is there a SKOS predicate I can map to in case where predicate is unknown? I think most of these
#   mappings are attempts at exact matches, but I can't be sure (at least not without using these fields
#   to determine: mapGroup, mapPriority, mapRule, mapAdvice).
# mapCategoryId,mapCategoryName: Only these in set: 447637006 "MAP SOURCE CONCEPT IS PROPERLY CLASSIFIED",
#   447638001 "MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA",
#   447639009 "MAP OF SOURCE CONCEPT IS CONTEXT DEPENDENT"
# 'predicate_modifier': '???',
#   Description: Modifier for negating the prediate. See https://github.com/mapping-commons/sssom/issues/40
#   Range: PredicateModifierEnum: (joe: only lists 'Not' as an option)
#   Example: Not Negates the predicate, see documentation of predicate_modifier_enum
# - predicate_id <- mapAdvice?
# - predicate_modifier <- mapAdvice?
#   mapAdvice: Pipe-delimited qualifiers. Ex:
#   "ALWAYS Q71.30 | CONSIDER LATERALITY SPECIFICATION"
#   "IF LISSENCEPHALY TYPE 3 FAMILIAL FETAL AKINESIA SEQUENCE SYNDROME CHOOSE Q04.3 | MAP OF SOURCE CONCEPT
#   IS CONTEXT DEPENDENT"
#   "MAP SOURCE CONCEPT CANNOT BE CLASSIFIED WITH AVAILABLE DATA"
'predicate_id': f'SNOMED:{row["mapCategoryId"]}',
'predicate_label': row['mapCategoryName'],

'object_id': f'ICD10CM:{row["mapTarget"]}',
'object_label': row['mapTargetName'],

# match_type <- mapRule?
#   ex: TRUE: when "ALWAYS <code>" is in pipe-delimited list in mapAdvice, this always shows TRUE. Does this
#       mean I could use skos:exactMatch in these cases?
# match_type <- correlationId?: This may look redundant, but I want to be explicit. In officially downloaded
#   SNOMED mappings, all of them had correlationId of 447561005, which also happens to be 'unspecified'.
#   If correlationId is indeed more appropriate for predicate_id, then I don't think there is a representative
#   field for 'match_type'.
'match_type': MatchTypeEnum('Unspecified') if row['correlationId'] == match_type_snomed_unspecified_id \
    else  MatchTypeEnum('Unspecified'),

'mapping_date': date_parser.parse(str(row['effectiveTime'])).date(),
'other': '|'.join([f'{k}={str(row[k])}' for k in [
    'id',
    'active',
    'moduleId',
    'refsetId',
    'mapGroup',
    'mapPriority',
    'mapRule',
    'mapAdvice',
]]),

# More fields (https://mapping-commons.github.io/sssom/Mapping/):
# - subject_category: absent
# - author_id: can this be "SNOMED"?
# - author_label: can this be "SNOMED"?
# - reviewer_id: can this be "SNOMED"?
# - reviewer_label: can this be "SNOMED"?
# - creator_id: can this be "SNOMED"?
# - creator_label: can this be "SNOMED"?
# - license: Is this something that can be determined?
# - subject_source: URL of some official page for SNOMED version used?
# - subject_source_version: Is this knowable?
# - objectCategory <= mapRule?
#   mapRule: ex: TRUE: when "ALWAYS <code>" is in pipe-delimited list in mapAdvice, this always shows TRUE.
#     Does this mean I could use skos:exactMatch in these cases?
#     object_category:
#   objectCategory:
#     Description: The conceptual category to which the subject belongs to. This can be a string denoting
#     the category or a term from a controlled vocabulary.
#     Example: UBERON:0001062 (The CURIE of the Uberon term for "anatomical entity".)
# - object_source: URL of some official page for ICD10CM version used?
# - object_source_version: would this be "10CM" as in "ICD10CM"? Or something else? Or nothing?
# - mapping_provider: can this be "SNOMED"?
# - mapping_cardinality: Could I determine 1:1 or 1:many or many:1 based on:
#   mapGroup, mapPriority, mapRule, mapAdvice?
# - match_term_type: What is this?
# - see_also: Should this be a URL to the SNOMED term?
# - comment: Description: Free text field containing either curator notes or text generated by tool providing
#   additional informative information.


SNOMED mapping related codes
----------------------------
match_type_snomed_unspecified_id = 447561005
https://www.findacode.com/snomed/447561005--snomed-ct-source-code-to-target-map-correlation-not-specified.html

Additional resources
--------------------
About SNOMED simple and complex refsets:
https://github.com/HOT-Ecosystem/tccm/blob/master/docs/SNOMED/MapRefsets.md
