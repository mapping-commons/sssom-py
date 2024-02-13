.. _mapping_set_document:

MappingSetDocument
==================

.. module:: Additional SSSOM object models
   :synopsis: This module provides a data structure to represent a single SSSOM document.

.. class:: MappingSetDocument
   :noindex:

    A data class representing a single SSSOM document which is a holder for a MappingSet object plus a CURIE map.

    .. attribute:: mapping_set

        The main part of the document: a set of mappings plus metadata. It is of type :class:`sssom_schema.MappingSet`.

    .. attribute:: converter

        An instance of :class:`curies.Converter`, used to convert CURIEs in the mapping set.

    .. automethod:: prefix_map

        A method that returns a prefix map, which is a dictionary representation of the converter's bidirectional map. 
        The returned dictionary is of type :class:`typing.Dict` where both keys and values are of type :class:`str`.
