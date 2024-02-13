.. py:module:: <module_name>
.. py:function:: rewire_graph(g: Graph, mset: MappingSetDataFrame, subject_to_object: bool = True, precedence: Optional[List[str]] = None) -> int:

   This function rewrites an RDF Graph by replacing nodes using equivalence mappings provided in a MappingSetDataFrame. The function can either map from subject to object (default behavior) or from object to subject. In case of ambiguity, precedence can be defined with a list of prefixes.

   :param g: The RDF graph to be rewired.
   :type g: Graph
   :param mset: The MappingSetDataFrame containing the equivalence mappings.
   :type mset: MappingSetDataFrame
   :param subject_to_object: A boolean that determines the direction of the rewire. If true, the rewire is from subject to object. If false, the rewire is from object to subject. Default is True.
   :type subject_to_object: bool, optional
   :param precedence: A list of prefixes to resolve any ambiguity in the mapping. If not provided, the function raises a ValueError when it encounters ambiguity. Default is None.
   :type precedence: list, optional
   :returns: The number of triples in the graph that were changed.
   :rtype: int
   :raises TypeError: If a mapping does not contain a valid subject or object.
   :raises ValueError: If there is ambiguity in the mapping and no precedence is defined.

.. py:function:: rewire_node(n: Any)

   This function is a helper function used to rewire individual nodes in the graph.

   :param n: The node in the graph to be rewired.
   :type n: Any
   :returns: The rewired node. If the node is not in the rewire map, it returns the original node.
   :rtype: Any
