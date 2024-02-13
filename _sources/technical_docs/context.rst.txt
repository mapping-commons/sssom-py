Utilities for loading JSON-LD contexts
======================================

This module provides utility functions for loading JSON-LD contexts.

Functions
---------

.. autofunction:: get_converter
.. autofunction:: ensure_converter

Variables and Constants
-----------------------

.. autodata:: SSSOM_BUILT_IN_PREFIXES
.. autodata:: SSSOM_CONTEXT
.. autodata:: ConverterHint

Private Functions
-----------------

.. autofunction:: _get_default_converter
.. autofunction:: _load_sssom_context
.. autofunction:: _get_built_in_prefix_map

Details
-------

**get_converter**

This function returns a converter. The converter is obtained by chaining a built-in prefix map with a default converter. The function is cached using `functools.lru_cache` which means the result is saved and returned for any subsequent calls without re-executing the function.

**_get_default_converter**

This private function returns a converter from an extended prefix map. It checks each record in the converter, keeps only those with a valid NCName (valid names for elements and attributes), and then returns a new Converter with these records.

**_load_sssom_context**

This private function loads the JSON-LD context from a file and returns it as a dictionary.

**_get_built_in_prefix_map**

This private function returns a converter for built-in prefixes. The converter is created from prefixes in the JSON-LD context that match the built-in prefixes. The function is also cached using `functools.lru_cache`.

**ensure_converter**

This function ensures a converter is available. The type of the converter depends on the `prefix_map` parameter. If `prefix_map` is None or an empty dictionary, and `use_defaults` is set to True, it returns the default converter. If `prefix_map` is a non-empty dictionary, it is converted into a converter and chained behind the built-in prefix map. If `prefix_map` is a pre-instantiated converter, it is also chained behind the built-in prefix map.

**ConverterHint**

This type hint specifies a place where one of three options can be given:

1. None, which means a default converter is loaded.
2. A legacy prefix mapping dictionary which will get upgraded into a Converter.
3. A Converter which might get modified. 

**SSSOM_BUILT_IN_PREFIXES**

This variable contains a tuple of built-in prefixes.

**SSSOM_CONTEXT**

This variable is the path to the JSON-LD context file.

