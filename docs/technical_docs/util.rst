## Utility functions

### `augment_metadata`

Augments metadata with parameters passed.

```
augment_metadata(msdf: MappingSetDataFrame, meta: dict, replace_multivalued: bool = False) -> MappingSetDataFrame
```

#### Arguments

- `msdf`: MappingSetDataFrame (MSDF) object.
- `meta`: Dictionary that needs to be added/updated to the metadata of the MSDF.
- `replace_multivalued`: Multivalued slots should be
    replaced or not, defaults to False.

#### Raises

- `ValueError`: If type of slot is neither str nor list.

#### Returns

MSDF with updated metadata.

---

### `compare_dataframes`

Perform a diff between two SSSOM dataframes.

```
compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> MappingSetDiff
```

#### Arguments

- `df1`: A mapping dataframe
- `df2`: A mapping dataframe

#### Returns

A mapping set diff

.. warning:: currently does not discriminate between mappings with different predicates

---

### `create_entity`

Create an Entity object.

```
create_entity(identifier: str, mappings: Dict[str, Any]) -> Uriorcurie
```

#### Arguments

- `identifier`: Entity Id
- `mappings`: Mapping dictionary

#### Returns

An Entity object

---

### `deal_with_negation`

Combine negative and positive rows with matching [SUBJECT_ID, OBJECT_ID, CONFIDENCE] combination.

Rule: negative trumps positive if modulus of confidence values are equal.

```
deal_with_negation(df: pd.DataFrame) -> pd.DataFrame
```

#### Arguments

- `df`: Merged Pandas DataFrame

#### Returns

Pandas DataFrame with negations addressed

#### Raises

- `ValueError`: If the dataframe is none after assigning default confidence

---

### `filter_out_prefixes`

Filter out rows which contains a CURIE with a prefix in the filter_prefixes list.

```
filter_out_prefixes(df: pd.DataFrame, filter_prefixes: List[str], features: Optional[list] = None, require_all_prefixes: bool = False) -> pd.DataFrame
```

#### Arguments

- `df`: Pandas DataFrame of SSSOM Mapping
- `filter_prefixes`: List of prefixes
- `features`: List of dataframe column names dataframe to consider
- `require_all_prefixes`: If True, all prefixes must be present in a row to be filtered out

#### Returns

Pandas Dataframe

---

### `filter_prefixes`

Filter out rows which do NOT contain a CURIE with a prefix in the filter_prefixes list.

```
filter_prefixes(df: pd.DataFrame, filter_prefixes: List[str], features: Optional[list] = None, require_all_prefixes: bool = True) -> pd.DataFrame
```

#### Arguments

- `df`: Pandas DataFrame of SSSOM Mapping
- `filter_prefixes`: List of prefixes
- `features`: List of dataframe column names dataframe to consider
- `require_all_prefixes`: If True, all prefixes must be present in a row to be filtered out

#### Returns

Pandas Dataframe

---

### `get_all_prefixes`

Fetch all prefixes in the MappingSetDataFrame.

```
get_all_prefixes(msdf: MappingSetDataFrame) -> Set[str]
```

#### Arguments

- `msdf`: MappingSetDataFrame

#### Raises

- `ValidationError`: If slot is wrong.
- `ValidationError`: If slot is wrong.

#### Returns

List of all prefixes.

---

### `get_dict_from_mapping`

Get information for linkml objects (MatchTypeEnum, PredicateModifierEnum) from the Mapping object and return the dictionary form of the object.

```
get_dict_from_mapping(map_obj: Union[Any, Dict[Any, Any], SSSOM_Mapping]) -> dict
```

#### Arguments

- `map_obj`: Mapping object

#### Returns

Dictionary

---

### `get_file_extension`

Get file extension.

```
get_file_extension(file: Union[str, Path, TextIO]) -> str
```

#### Arguments

- `file`: File path

#### Returns

format of the file passed, default tsv

---

### `get_prefixes_used_in_metadata`

Get a set of prefixes used in CURIEs in the metadata.

```
get_prefixes_used_in_metadata(meta: MetadataType) -> Set[str]
```

#### Arguments

- `meta`: MetadataType

#### Returns

Set of prefixes

---

### `get_prefixes_used_in_table`

Get a list of prefixes used in CURIEs in key feature columns in a dataframe.

```
get_prefixes_used_in_table(df: pd.DataFrame, converter: Converter) -> Set[str]
```

#### Arguments

- `df`: Pandas DataFrame of SSSOM Mapping
- `converter`: Converter

#### Returns

Set of prefixes

---

### `group_mappings`

Group mappings by EntityPairs.

```
group_mappings(df: pd.DataFrame) -> Dict[EntityPair, List[pd.Series]]
```

#### Arguments

- `df`: DataFrame whose `confidence` column needs to be filled.

#### Returns

Dictionary consisting of the original DataFrame and dataframe consisting of empty confidence values.

---

### `inject_metadata_into_df`

Inject metadata dictionary key-value pair into DataFrame columns in a MappingSetDataFrame.DataFrame.

```
inject_metadata_into_df(msdf: MappingSetDataFrame) -> MappingSetDataFrame
```

#### Arguments

- `msdf`: MappingSetDataFrame with metadata separate.

#### Returns

MappingSetDataFrame with metadata as columns

---

### `invert_mappings`

Switching subject and objects based on their prefixes and adjusting predicates accordingly.

```
invert_mappings(df: pd.DataFrame, subject_prefix: Optional[str] = None, merge_inverted: bool = True, predicate_invert_dictionary: dict = None) -> pd.DataFrame
```

#### Arguments

- `df`: Pandas dataframe.
- `subject_prefix`: Prefix of subjects desired.
- `merge_inverted`: If True (default), add inverted dataframe to input else,
                      just return inverted data.
- `predicate_invert_dictionary`: YAML file providing the inverse mapping for predicates.

#### Returns

Pandas dataframe with all subject IDs having the same prefix.

---

### `is_multivalued_slot`

Check whether the slot is multivalued according to the SSSOM specification.

```
is_multivalued_slot(slot: str) -> bool
```

#### Arguments

- `slot`: Slot name

#### Returns

Slot is multivalued or no

---

### `merge_msdf`

Merge multiple MappingSetDataFrames into one.

```
merge_msdf(*msdfs: MappingSetDataFrame, reconcile: bool = False) -> MappingSetDataFrame
```

#### Arguments

- `msdfs`: A Tuple of MappingSetDataFrames to be merged
- `reconcile`: If reconcile=True, then dedupe(remove redundant lower confidence mappings)
    and reconcile (if msdf contains a higher confidence _negative_ mapping,
    then remove lower confidence positive one. If confidence is the same,
    prefer HumanCurated. If both HumanCurated, prefer negative mapping).
    Defaults to True.

#### Returns

Merged MappingSetDataFrame.

---

### `parse`

Parse a TSV to a pandas frame.

```
parse(filename: Union[str, Path]) -> pd.DataFrame
```

#### Arguments

- `filename`: Filename or filepath

#### Returns

Pandas DataFrame

---

### `raise_for_bad_path`

Raise exception if file path is invalid.

```
raise_for_bad_path(file_path: Union[str, Path]) -> None
```

#### Arguments

- `file_path`: File path

#### Raises

- `FileNotFoundError`: Invalid file path

---

### `reconcile_prefix_and_data`

Reconciles prefix_map and translates CURIE switch in dataframe.

```
reconcile_prefix_and_data(msdf: MappingSetDataFrame, prefix_reconciliation: dict) -> MappingSetDataFrame
```

#### Arguments

- `msdf`: Mapping Set DataFrame.
- `prefix_reconciliation`: Prefix reconcilation dictionary from a YAML file

#### Returns

Mapping Set DataFrame with reconciled prefix_map and data.

---

### `safe_compress`

Parse a CURIE from an IRI.

```
safe_compress(uri: str, converter: Converter) -> str
```

#### Arguments

- `uri`: The URI to parse. If this is already a CURIE, return directly.
- `converter`: Converter used for compression

#### Returns

A CURIE

---

### `sort_df_rows_columns`

Canonical sorting of DataFrame columns.

```
sort_df_rows_columns(df: pd.DataFrame, by_columns: bool = True, by_rows: bool = True) -> pd.DataFrame
```

#### Arguments

- `df`: Pandas DataFrame with random column sequence.
- `by_columns`: Boolean flag to sort columns canonically.
- `by_rows`: Boolean flag to sort rows by column #1