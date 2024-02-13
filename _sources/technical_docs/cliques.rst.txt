Utilities for identifying and working with cliques/SCCs in mappings graphs
--------------------------------------------------------------------------

This module contains several utilities for working with strongly connected components (SCCs) or cliques in mapping graphs.
These utilities are used for identifying and manipulating SCCs in the context of data mappings.

Functions
---------

.. function:: to_digraph(msdf: MappingSetDataFrame) -> "networkx.DiGraph"

   Convert a MappingSetDataFrame to a directed graph where the nodes are entities' CURIEs and edges are their mappings.

   :param msdf: A MappingSetDataFrame object.
   :return: A networkx.DiGraph object.

.. function:: split_into_cliques(msdf: MappingSetDataFrame) -> List[MappingSetDocument]

   Split a MappingSetDataFrame into documents corresponding to strongly connected components of the associated graph.

   :param msdf: A MappingSetDataFrame object.
   :return: List of MappingSetDocument objects representing strongly connected components.
   :raises TypeError: If Mappings is not of type List.

.. function:: group_values(d: Dict[str, str]) -> Dict[str, List[str]]

   Group all keys in the dictionary that share the same value.

   :param d: A dictionary.
   :return: A dictionary where values are grouped by keys.

.. function:: get_src(src: Optional[str], curie: str) -> str

   Get the prefix of a subject/object in the MappingSetDataFrame.

   :param src: Source.
   :param curie: CURIE.
   :return: Source prefix.

.. function:: summarize_cliques(doc: MappingSetDataFrame) -> pd.DataFrame

   Summarize statistics on a clique document.

   :param doc: A MappingSetDataFrame object.
   :return: A DataFrame containing summary statistics for each clique.
