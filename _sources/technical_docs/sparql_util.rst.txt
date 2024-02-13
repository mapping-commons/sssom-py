```
The module provides utilities for querying mappings from a SPARQL endpoint. It uses the SPARQLWrapper to interact with the SPARQL endpoint and pandas library for data manipulation. It also uses the RDFLib library for handling RDF.

Classes:

- EndpointConfig: A container for a SPARQL endpoint's configuration.

Functions:

- query_mappings: Queries a SPARQL endpoint to obtain a set of mappings.

Classes:
========

.. autoclass:: EndpointConfig
    :members:

    This dataclass represents the configuration for a SPARQL endpoint. It has the following attributes:

    - url: The URL of the SPARQL endpoint.
    - graph: The URI reference for the graph in the RDF store.
    - converter: An instance of curies.Converter to convert CURIEs to URIs.
    - predmap: A dictionary mapping predicates to strings.
    - predicates: A list of predicates to be queried. If not provided, defaults to SKOS.exactMatch and SKOS.closeMatch.
    - limit: An optional limit on the number of results returned by the query.
    - include_object_labels: A flag indicating whether to include labels for objects in the query results.

Functions:
==========

.. autofunction:: query_mappings

    This function takes an EndpointConfig instance as an argument and returns a MappingSetDataFrame containing the query results. It constructs a SPARQL query based on the configuration and executes it against the endpoint. If the 'include_object_labels' flag is set, the query includes labels for the objects. The function logs the SPARQL query and converts the results into a pandas DataFrame.

```
