Type: mapping set
=================

URI: `sssom:MappingSet <http://w3id.org/sssom/MappingSet>`__

.. figure:: http://yuml.me/diagram/nofunky;dir:TB/class/%5BEntity%5D%3Cobject_match_field%200..1-++%5BMappingSet%7Cmapping_set_version:string%20%3F;creator_label:string%20%3F;license:string%20%3F;subject_source:string%20%3F;subject_source_version:string%20%3F;object_source:string%20%3F;object_source_version:string%20%3F;mapping_provider:string%20%3F;mapping_tool:string%20%3F;mapping_date:string%20%3F;subject_preprocessing:string%20%3F;object_preprocessing:string%20%3F;match_term_type:string%20%3F;see_also:string%20%3F;other:string%20%3F;comment:string%20%3F%5D,%20%5BEntity%5D%3Csubject_match_field%200..1-++%5BMappingSet%5D,%20%5BEntity%5D%3Ccreator_id%200..1-++%5BMappingSet%5D,%20%5BEntity%5D%3Cmapping_set_id%200..1-++%5BMappingSet%5D,%20%5BMapping%5D%3Cmappings%200..*-++%5BMappingSet%5D
   :alt: img

   img

Attributes
----------

Own
~~~

-  `mapping_set_id <mapping_set_id.md>`__ OPT

   -  Description: A globally unique identifier for the mapping set (not
      each individual mapping). Should be IRI, ideally resolvable.
   -  range: `Entity <Entity.md>`__

-  `mapping_set_version <mapping_set_version.md>`__ OPT

   -  Description: A version string for the mapping.
   -  range: `String <types/String.md>`__

-  `mappings <mappings.md>`__ 0..\*

   -  range: `Mapping <Mapping.md>`__

Inherited from mapping:
~~~~~~~~~~~~~~~~~~~~~~~

-  `subject_id <subject_id.md>`__ OPT

   -  Description: The ID of the subject of the mapping.
   -  range: `Entity <Entity.md>`__
   -  inherited from: None

-  `subject_label <subject_label.md>`__ OPT

   -  Description: The label of subject of the mapping
   -  range: `String <types/String.md>`__
   -  inherited from: None

-  `predicate_id <predicate_id.md>`__ OPT

   -  Description: The ID of the predicate or relation that relates the
      subject and object of this match.
   -  range: `Entity <Entity.md>`__
   -  inherited from: None

-  `predicate_label <predicate_label.md>`__ OPT

   -  Description: The label of the predicate/relation of the mapping
   -  range: `String <types/String.md>`__
   -  inherited from: None

-  `object_id <object_id.md>`__ OPT

   -  Description: The ID of the object of the mapping.
   -  range: `Entity <Entity.md>`__
   -  inherited from: None

-  `object_label <object_label.md>`__ OPT

   -  Description: The label of object of the mapping
   -  range: `String <types/String.md>`__
   -  inherited from: None

-  `match_type <match_type.md>`__ OPT

   -  Description: ID from Match type (SSSOM:MatchType) branch of the
      SSSSOM Vocabulary. In the case of multiple match types for a
      single subject, predicate, object triplet, two seperate mappings
      must be specified.
   -  range: `String <types/String.md>`__
   -  inherited from: None

-  `creator_id <creator_id.md>`__ OPT

   -  Description: Identifies the persons or groups responsible for the
      creation of the mapping. Recommended to be a (pipe-separated) list
      of ORCIDs or otherwise identifying URLs, but any identifying
      string (such as name and affiliation) is permissible.
   -  range: `Entity <Entity.md>`__

-  `creator_label <creator_label.md>`__ OPT

   -  Description: A string identifying the creator of this mapping. In
      the spirit of provenance, consider to use creator_id instead.
   -  range: `String <types/String.md>`__

-  `license <license.md>`__ OPT

   -  Description: A url to the license of the mapping. In absence of a
      license we assume no license.
   -  range: `String <types/String.md>`__

-  `subject_source <subject_source.md>`__ OPT

   -  Description: IRI of ontology source for the subject. Version IRI
      preferred.
   -  range: `String <types/String.md>`__

-  `subject_source_version <subject_source_version.md>`__ OPT

   -  Description: Version IRI of the source of the subject term.
   -  range: `String <types/String.md>`__

-  `object_source <object_source.md>`__ OPT

   -  Description: IRI of ontology source for the object. Version IRI
      preferred.
   -  range: `String <types/String.md>`__

-  `object_source_version <object_source_version.md>`__ OPT

   -  Description: Version IRI of the source of the object term.
   -  range: `String <types/String.md>`__

-  `mapping_provider <mapping_provider.md>`__ OPT

   -  Description: URL pointing to the source that provided the mapping,
      for example an ontology that already contains the mappings.
   -  range: `String <types/String.md>`__

-  `mapping_tool <mapping_tool.md>`__ OPT

   -  Description: A reference to the tool or algorithm that was used to
      generate the mapping. Should be a URL pointing to more info about
      it, but can be free text.
   -  range: `String <types/String.md>`__

-  `mapping_date <mapping_date.md>`__ OPT

   -  Description: The date the mapping was computed
   -  range: `String <types/String.md>`__

-  `confidence <confidence.md>`__ OPT

   -  Description: A score between 0 and 1 to denote the confidence or
      probability that the match is correct, where 1 denotes total
      confidence.
   -  range: `Double <types/Double.md>`__
   -  inherited from: None

-  `subject_match_field <subject_match_field.md>`__ OPT

   -  Description: A tuple of fields (term annotations on the subject)
      that was used for the match. Should be used in conjunction with
      lexical and complexes matches, see SSSOM match types below.
   -  range: `Entity <Entity.md>`__

-  `object_match_field <object_match_field.md>`__ OPT

   -  Description: A tuple of fields (term annotations on the object)
      that was used for the match. Should be used in conjunction with
      lexical and complexes matches, see SSSOM match types below.
   -  range: `Entity <Entity.md>`__

-  `match_string <match_string.md>`__ OPT

   -  Description: String that is shared by subj/obj
   -  range: `String <types/String.md>`__
   -  inherited from: None

-  `subject_preprocessing <subject_preprocessing.md>`__ OPT

   -  Description: Method of preprocessing applied to the fields of the
      subject. Tuple of IDs from “Pre-processing method”
      (SSSOM:PreprocessingMethod) branch of the SSSSOM Vocabulary.
   -  range: `String <types/String.md>`__

-  `object_preprocessing <object_preprocessing.md>`__ OPT

   -  Description: Method of preprocessing applied to the fields of the
      object. Tuple of IDs from “Pre-processing method”
      (SSSOM:PreprocessingMethod) branch of the SSSSOM Vocabulary.
   -  range: `String <types/String.md>`__

-  `match_term_type <match_term_type.md>`__ OPT

   -  Description: Specifies what type of terms are being matched
      (class, property, or individual). Value should be ID from Term
      Match (SSSOM:TermMatch) branch of the SSSSOM Vocabulary.
   -  range: `String <types/String.md>`__

-  `semantic_similarity_score <semantic_similarity_score.md>`__ OPT

   -  Description: A score between 0 and 1 to denote the semantic
      similarity, where 1 denotes equivalence.
   -  range: `Double <types/Double.md>`__
   -  inherited from: None

-  `information_content_mica_score <information_content_mica_score.md>`__
   OPT

   -  Description: A score between 0 and 1 to denote the information
      content of the most informative common ancestor, where 1 denotes
      the maximum level of informativeness.
   -  range: `Double <types/Double.md>`__
   -  inherited from: None

-  `see_also <see_also.md>`__ OPT

   -  Description: A URL specific for the mapping instance. E.g. for
      kboom we have a per-mapping image that shows surrounding axioms
      that drive probability. Could also be a github issue URL that
      discussed a complicated alignment
   -  range: `String <types/String.md>`__

-  `other <other.md>`__ OPT

   -  Description: Pipe separated list of key value pairs for properties
      not part of the SSSOM spec. Can be used to encode additional
      provenance data.
   -  range: `String <types/String.md>`__

-  `comment <comment.md>`__ OPT

   -  Description: Free text field containing either curator notes or
      text generated by tool providing additional informative
      information.
   -  range: `String <types/String.md>`__
