```
.. module:: Validators
   :synopsis: A module for validating various schema types.

.. autoclass:: validate
   :members:

.. autofunction:: validate
   :param msdf: MappingSetDataFrame.
   :param validation_types: SchemaValidationType
   :return: None
   :rtype: None

.. autofunction:: validate_json_schema
   :param msdf: MappingSetDataFrame to be validated.
   :return: None
   :rtype: None 

.. autofunction:: validate_shacl
   :param msdf: MappingSetDataFrame
   :raises NotImplementedError: Not yet implemented.
   :return: None
   :rtype: None 

.. autofunction:: validate_sparql
   :param msdf: MappingSetDataFrame
   :raises NotImplementedError: Not yet implemented.
   :return: None
   :rtype: None 

.. autofunction:: check_all_prefixes_in_curie_map
   :param msdf: MappingSetDataFrame
   :raises ValidationError: If all prefixes not in curie_map.
   :return: None
   :rtype: None 

.. data:: VALIDATION_METHODS
   :annotation: Mapping[SchemaValidationType, Callable]
   :value: A mapping of different validation methods for different schema types.
```