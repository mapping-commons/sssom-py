# sssom-cliquesummary

Data dictionary for clique summary

Slots:

.. list-table::
    :widths: 10 45 45
    :header-rows: 1

    * - Slot Name
        - URI
        - Range
    * - id
        - http://www.w3.org/2002/07/owl#id
        - http://www.w3.org/2001/XMLSchema#string
    * - members
        - http://w3id.org/sssom/schema/cliquesummary/members
        - http://www.w3.org/2001/XMLSchema#string
    * - members_labels
        - http://w3id.org/sssom/schema/cliquesummary/members_labels
        - http://www.w3.org/2001/XMLSchema#string
    * - num_members
        - http://w3id.org/sssom/schema/cliquesummary/num_members
        - http://www.w3.org/2001/XMLSchema#integer
    * - sources
        - http://w3id.org/sssom/schema/cliquesummary/sources
        - http://www.w3.org/2001/XMLSchema#string
    * - num_sources
        - http://w3id.org/sssom/schema/cliquesummary/num_sources
        - http://www.w3.org/2001/XMLSchema#integer
    * - max_confidence
        - http://w3id.org/sssom/schema/cliquesummary/max_confidence
        - http://www.w3.org/2001/XMLSchema#double
    * - min_confidence
        - http://w3id.org/sssom/schema/cliquesummary/min_confidence
        - http://www.w3.org/2001/XMLSchema#double
    * - avg_confidence
        - http://w3id.org/sssom/schema/cliquesummary/avg_confidence
        - http://www.w3.org/2001/XMLSchema#double
    * - is_conflated
        - http://w3id.org/sssom/schema/cliquesummary/is_conflated
        - http://www.w3.org/2001/XMLSchema#boolean
    * - is_all_conflated
        - http://w3id.org/sssom/schema/cliquesummary/is_all_conflated
        - http://www.w3.org/2001/XMLSchema#boolean
    * - total_conflated
        - http://w3id.org/sssom/schema/cliquesummary/total_conflated
        - http://www.w3.org/2001/XMLSchema#integer
    * - proportion_conflated
        - http://w3id.org/sssom/schema/cliquesummary/proportion_conflated
        - http://www.w3.org/2001/XMLSchema#double
    * - conflation_score
        - http://w3id.org/sssom/schema/cliquesummary/conflation_score
        - http://www.w3.org/2001/XMLSchema#double
    * - members_count
        - http://w3id.org/sssom/schema/cliquesummary/members_count
        - http://www.w3.org/2001/XMLSchema#integer
    * - min_count_by_source
        - http://w3id.org/sssom/schema/cliquesummary/min_count_by_source
        - http://www.w3.org/2001/XMLSchema#integer
    * - max_count_by_source
        - http://w3id.org/sssom/schema/cliquesummary/max_count_by_source
        - http://www.w3.org/2001/XMLSchema#integer
    * - avg_count_by_source
        - http://w3id.org/sssom/schema/cliquesummary/avg_count_by_source
        - http://www.w3.org/2001/XMLSchema#double
    * - harmonic_mean_count_by_source
        - http://w3id.org/sssom/schema/cliquesummary/harmonic_mean_count_by_source
        - http://www.w3.org/2001/XMLSchema#double