
# Class: entity


Represents any entity that can be mapped, such as an OWL class or SKOS concept

URI: [sssom:Entity](http://w3id.org/sssom/Entity)


[![img](https://yuml.me/diagram/nofunky;dir:TB/class/[MappingSet]-%20creator_id%200..1>[Entity&#124;id:string;label:string%20%3F;category:string%20%3F;source:string%20%3F],[Mapping]-%20creator_id%200..1>[Entity],[MappingSet]-%20mapping_set_id%200..1>[Entity],[Mapping]-%20object_id%200..1>[Entity],[MappingSet]-%20object_match_field%200..1>[Entity],[Mapping]-%20object_match_field%200..1>[Entity],[Mapping]-%20predicate_id%200..1>[Entity],[Mapping]-%20subject_id%200..1>[Entity],[MappingSet]-%20subject_match_field%200..1>[Entity],[Mapping]-%20subject_match_field%200..1>[Entity],[MappingSet],[Mapping])](https://yuml.me/diagram/nofunky;dir:TB/class/[MappingSet]-%20creator_id%200..1>[Entity&#124;id:string;label:string%20%3F;category:string%20%3F;source:string%20%3F],[Mapping]-%20creator_id%200..1>[Entity],[MappingSet]-%20mapping_set_id%200..1>[Entity],[Mapping]-%20object_id%200..1>[Entity],[MappingSet]-%20object_match_field%200..1>[Entity],[Mapping]-%20object_match_field%200..1>[Entity],[Mapping]-%20predicate_id%200..1>[Entity],[Mapping]-%20subject_id%200..1>[Entity],[MappingSet]-%20subject_match_field%200..1>[Entity],[Mapping]-%20subject_match_field%200..1>[Entity],[MappingSet],[Mapping])

## Referenced by class

 *  **None** *[creator_id](creator_id.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[mapping_set_id](mapping_set_id.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[object_id](object_id.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[object_match_field](object_match_field.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[predicate_id](predicate_id.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[subject_id](subject_id.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**
 *  **None** *[subject_match_field](subject_match_field.md)*  <sub>0..1</sub>  **[Entity](Entity.md)**

## Attributes


### Own

 * [id](id.md)  <sub>1..1</sub>
     * Description: CURIE or IRI identifier
     * Range: [String](types/String.md)
 * [label](label.md)  <sub>0..1</sub>
     * Description: label of an entity
     * Range: [String](types/String.md)
 * [category](category.md)  <sub>0..1</sub>
     * Description: category of the entity. Could be biolink, COB, etc
     * Range: [String](types/String.md)
 * [source](source.md)  <sub>0..1</sub>
     * Description: the database or ontology prefix of the entity
     * Range: [String](types/String.md)

## Other properties

|  |  |  |
| --- | --- | --- |
| **Mappings:** | | rdf:Resource |

