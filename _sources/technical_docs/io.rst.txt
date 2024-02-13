## I/O Utilities for SSSOM

This module contains I/O utility functions for working with SSSOM files and dataframes.

### Functions

#### `convert_file()`

Convert a file from one format to another.

```
def convert_file(
    input_path: str,
    output: TextIO,
    output_format: Optional[str] = None,
) -> None:
```

**Arguments:**

* `input_path`: The path to the input SSSOM tsv file
* `output`: The path to the output file. If none is given, will default to using stdout.
* `output_format`: The format to which the SSSOM TSV should be converted.

#### `parse_file()`

Parse an SSSOM metadata file and write to a table.

```
def parse_file(
    input_path: str,
    output: TextIO,
    *,
    input_format: Optional[str] = None,
    metadata_path: Optional[str] = None,
    prefix_map_mode: Optional[MergeMode] = None,
    clean_prefixes: bool = True,
    strict_clean_prefixes: bool = True,
    embedded_mode: bool = True,
    mapping_predicate_filter: tuple = None,
) -> None:
```

**Arguments:**

* `input_path`: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
* `output`: The path to the output file.
* `input_format`: The string denoting the input format.
* `metadata_path`: The path to a file containing the sssom metadata (including prefix_map)
        to be used during parse.
* `prefix_map_mode`: Defines whether the prefix map in the metadata should be extended or replaced with
        the SSSOM default prefix map derived from the :mod:`bioregistry`.
* `clean_prefixes`: If True (default), records with unknown prefixes are removed from the SSSOM file.
* `strict_clean_prefixes`: If True (default), clean_prefixes() will be in strict mode.
* `embedded_mode`: If True (default), the dataframe and metadata are exported in one file (tsv), else two separate files (tsv and yaml).
* `mapping_predicate_filter`: Optional list of mapping predicates or filepath containing the same.

#### `validate_file()`

Validate the incoming SSSOM TSV according to the SSSOM specification.

```
def validate_file(input_path: str, validation_types: List[SchemaValidationType]) -> None:
```

**Arguments:**

* `input_path`: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
* `validation_types`: A list of validation types to run.

#### `split_file()`

Split an SSSOM TSV by prefixes and relations.

```
def split_file(input_path: str, output_directory: Union[str, Path]) -> None:
```

**Arguments:**

* `input_path`: The path to the input file in one of the legal formats, eg obographs, aligmentapi-xml
* `output_directory`: The directory to which the split file should be exported.

#### `get_metadata_and_prefix_map()`

Load metadata and a prefix map in a deprecated way.

```
@deprecated(
    deprecated_in="0.4.3",
    details="This functionality for loading SSSOM metadata from a YAML file is deprecated from the "
    "public API since it has internal assumptions which are usually not valid for downstream users.",
)
def get_metadata_and_prefix_map(
    metadata_path: Union[None, str, Path] = None, *, prefix_map_mode: Optional[MergeMode] = None
) -> Tuple[Converter, MetadataType]:
```

**Arguments:**

* `metadata_path`: The metadata file in YAML format
* `prefix_map_mode`: one of metadata_only, sssom_default_only, merged

#### `extract_iris()`

Recursively extracts a list of IRIs from a string or file.

```
def extract_iris(
    input: Union[str, Path, Iterable[Union[str, Path]]], converter: Converter
) -> List[str]:
```

**Arguments:**

* `input`: CURIE OR list of CURIEs OR file path containing the same.
* `converter`: Prefix map of mapping set (possibly) containing custom prefix:IRI combination.

#### `filter_file()`

Filter a dataframe by dynamically generating queries based on user input.

```
def filter_file(input: str, output: Optional[TextIO] = None, **kwargs) -> MappingSetDataFrame:
```

**Arguments:**

* `input`: DataFrame to be queried over.
* `output`: Output location.
* `kwargs`: Filter options provided by user which generate queries (e.g.: --subject_id x:%).

#### `annotate_file()`

Annotate a file i.e. add custom metadata to the mapping set.

```
def annotate_file(
    input: str, output: Optional[TextIO] = None, replace_multivalued: bool = False, **kwargs
) -> MappingSetDataFrame:
```

**Arguments:**

* `input`: SSSOM tsv file to be queried over.
* `output`: Output location.
* `replace_multivalued`: Multivalued slots should be
        replaced or not, defaults to False
* `kwargs`: Options provided by user
        which are added to the metadata (e.g.: --mapping_set_id http://example.org/abcd)