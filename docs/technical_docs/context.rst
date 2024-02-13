.. _context:

JSON-LD context-loading utilities.
==================================

.. py:module:: sssom.context
   :synopsis: Utilities for loading JSON-LD contexts.

This module contains utility functions for loading JSON-LD contexts.

Functions
---------

.. py:function:: get_converter() -> Converter
   :module: sssom.context

   Get a converter that is used to convert prefixes into URIs. This function is decorated 
   with the `functools.lru_cache` decorator, which means that the function's results are 
   cached for faster subsequent calls. 

.. py:function:: ensure_converter(prefix_map: ConverterHint = None, *, use_defaults: bool = True) -> Converter
   :module: sssom.context

   Ensure a converter is available. This function checks if a converter is provided. If not, 
   it either returns the default converter (if `use_defaults` is `True`) or a converter 
   built from the built-in prefix map. If a converter or a prefix map is provided, it chains 
   this with the built-in prefix map and returns the result.

   :param prefix_map: One of three options can be given:

      1. An empty dictionary or ``None``. This results in using the default extended prefix 
         map (currently based on a variant of the Bioregistry) if ``use_defaults`` is set to 
         true, otherwise just the builtin prefix map including the prefixes in 
         :data:`SSSOM_BUILT_IN_PREFIXES`.
      2. A non-empty dictionary representing a prefix map. This is loaded as a converter with 
         :meth:`Converter.from_prefix_map`. It is chained behind the builtin prefix map to 
         ensure none of the :data:`SSSOM_BUILT_IN_PREFIXES` are overwritten with non-default 
         values.
      3. A pre-instantiated :class:`curies.Converter`. Similarly to a prefix map passed into 
         this function, this is chained behind the builtin prefix map.
   :param use_defaults: If an empty dictionary or None is passed to this function, this 
      parameter chooses if the extended prefix map (currently based on a variant of the 
      Bioregistry) gets loaded.
   :returns: A re-usable converter.

Constants
---------

.. py:data:: SSSOM_BUILT_IN_PREFIXES
   :module: sssom.context

   A tuple containing the names of the built-in prefixes that are used by SSSOM.

Types
-----

.. py:class:: ConverterHint
   :module: sssom.context

   A type hint that specifies a place where one of three options can be given:

      1. a legacy prefix mapping dictionary can be given, which will get upgraded into a 
         :class:`curies.Converter`,
      2. a converter can be given, which might get modified. In SSSOM-py, this typically 
         means chaining behind the "default" prefix map
      3. None, which means a default converter is loaded
