"""Serialization functions for SSSOM."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import pandas as pd
import yaml
from jsonasobj2 import JsonObj
from linkml_runtime.dumpers import JSONDumper, rdflib_dumper
from linkml_runtime.utils.schemaview import SchemaView
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF
from sssom.validators import check_all_prefixes_in_curie_map

# from .sssom_datamodel import slots
from sssom_schema import slots

from .constants import SCHEMA_YAML
from .parsers import to_mapping_set_document
from .typehints import PrefixMap
from .util import (
    PREFIX_MAP_KEY,
    RDF_FORMATS,
    SSSOM_DEFAULT_RDF_SERIALISATION,
    SSSOM_URI_PREFIX,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    get_file_extension,
    prepare_context_str,
    sort_df_rows_columns,
)

# from sssom.validators import check_all_prefixes_in_curie_map


# noinspection PyProtectedMember

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_ANNOTATION_PROPERTY = "http://www.w3.org/2002/07/owl#AnnotationProperty"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
OWL_EQUIV_OBJECTPROPERTY = "http://www.w3.org/2002/07/owl#equivalentProperty"
SSSOM_NS = SSSOM_URI_PREFIX

# Writers

MSDFWriter = Callable[[MappingSetDataFrame, TextIO], None]


def write_table(
    msdf: MappingSetDataFrame,
    file: TextIO,
    embedded_mode: bool = True,
    serialisation="tsv",
    sort=False,
) -> None:
    """Write a mapping set dataframe to the file as a table."""
    if msdf.df is None:
        raise TypeError

    sep = _get_separator(serialisation)

    # df = to_dataframe(msdf)

    meta: Dict[str, Any] = {}
    if msdf.metadata is not None:
        meta.update(msdf.metadata)
    if msdf.prefix_map is not None:
        meta[PREFIX_MAP_KEY] = msdf.prefix_map
    if sort:
        msdf.df = sort_df_rows_columns(msdf.df)
    lines = yaml.safe_dump(meta).split("\n")
    lines = [f"# {line}" for line in lines if line != ""]
    s = msdf.df.to_csv(sep=sep, index=False)

    if embedded_mode:
        lines = lines + [s]
        for line in lines:
            print(line, file=file)
    else:
        # Export MSDF as tsv
        msdf.df.to_csv(file, sep=sep, index=False)
        # Export Metadata as yaml
        yml_filepath = file.name.replace("tsv", "yaml")
        with open(yml_filepath, "w") as y:
            yaml.safe_dump(meta, y)


def write_rdf(
    msdf: MappingSetDataFrame,
    file: TextIO,
    serialisation: Optional[str] = None,
) -> None:
    """Write a mapping set dataframe to the file as RDF."""
    if serialisation is None:
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION
    elif serialisation not in RDF_FORMATS:
        logging.warning(
            f"Serialisation {serialisation} is not supported, "
            f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead."
        )
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    check_all_prefixes_in_curie_map(msdf)
    graph = to_rdf_graph(msdf=msdf)
    t = graph.serialize(format=serialisation, encoding="utf-8")
    print(t.decode(), file=file)


# todo: not sure the need for serialization param here; seems superfluous for some of these funcs
def write_fhir_json(
    msdf: MappingSetDataFrame, output: TextIO, serialisation="fhir"
) -> None:
    """Write a mapping set dataframe to the file as FHIR ConceptMap JSON."""
    data = to_fhir_json(msdf)
    json.dump(data, output, indent=2)


def write_json(msdf: MappingSetDataFrame, output: TextIO, serialisation="json") -> None:
    """Write a mapping set dataframe to the file as JSON."""
    if serialisation == "json":
        data = to_json(msdf)
        json.dump(data, output, indent=2)
    else:
        raise ValueError(
            f"Unknown json format: {serialisation}, currently only json supported"
        )


def write_owl(
    msdf: MappingSetDataFrame,
    file: TextIO,
    serialisation=SSSOM_DEFAULT_RDF_SERIALISATION,
) -> None:
    """Write a mapping set dataframe to the file as OWL."""
    if serialisation not in RDF_FORMATS:
        logging.warning(
            f"Serialisation {serialisation} is not supported, "
            f"using {SSSOM_DEFAULT_RDF_SERIALISATION} instead."
        )
        serialisation = SSSOM_DEFAULT_RDF_SERIALISATION

    graph = to_owl_graph(msdf)
    t = graph.serialize(format=serialisation, encoding="utf-8")
    print(t.decode(), file=file)


def write_ontoportal_json(
    msdf: MappingSetDataFrame, output: TextIO, serialisation="ontoportal_json"
) -> None:
    """Write a mapping set dataframe to the file as the ontoportal mapping JSON model."""
    if serialisation == "ontoportal_json":
        data = to_ontoportal_json(msdf)
        json.dump(data, output, indent=2)
    else:
        raise ValueError(
            f"Unknown json format: {serialisation}, currently only ontoportal_json supported"
        )


# Converters
# Converters convert a mappingsetdataframe to an object of the supportes types (json, pandas dataframe)


def to_dataframe(msdf: MappingSetDataFrame) -> pd.DataFrame:
    """Convert a mapping set dataframe to a dataframe."""
    data = []
    doc = to_mapping_set_document(msdf)
    if doc.mapping_set.mappings is None:
        raise TypeError
    for mapping in doc.mapping_set.mappings:
        mdict = mapping.__dict__
        m = {}
        for key in mdict:
            if mdict[key]:
                m[key] = mdict[key]
        data.append(m)
    df = pd.DataFrame(data=data)
    return df


def to_owl_graph(msdf: MappingSetDataFrame) -> Graph:
    """Convert a mapping set dataframe to OWL in an RDF graph."""
    graph = to_rdf_graph(msdf=msdf)

    for _s, _p, o in graph.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        graph.add((o, URIRef(RDF_TYPE), OWL.Axiom))

    for axiom in graph.subjects(RDF.type, OWL.Axiom):
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
                    graph.add((s, p, o))

    # if MAPPING_SET_ID in msdf.metadata:
    #    mapping_set_id = msdf.metadata[MAPPING_SET_ID]
    # else:
    #    mapping_set_id = DEFAULT_MAPPING_SET_ID

    sparql_prefixes = """
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX IAO: <http://purl.obolibrary.org/obo/IAO_>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

"""
    queries = []

    queries.append(
        sparql_prefixes
        + """
    INSERT {
      ?c rdf:type owl:Class .
      ?d rdf:type owl:Class .
    }
    WHERE {
     ?c owl:equivalentClass ?d .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
        INSERT {
          ?c rdf:type owl:ObjectProperty .
          ?d rdf:type owl:ObjectProperty .
        }
        WHERE {
         ?c owl:equivalentProperty ?d .
        }
        """
    )

    queries.append(
        sparql_prefixes
        + """
    DELETE {
      ?o rdf:type sssom:MappingSet .
    }
    INSERT {
      ?o rdf:type owl:Ontology .
    }
    WHERE {
     ?o rdf:type sssom:MappingSet .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
    DELETE {
      ?o sssom:mappings ?mappings .
    }
    WHERE {
     ?o sssom:mappings ?mappings .
    }
    """
    )

    queries.append(
        sparql_prefixes
        + """
    INSERT {
        ?p rdf:type owl:AnnotationProperty .
    }
    WHERE {
        ?o a owl:Axiom ;
        ?p ?v .
        FILTER(?p!=rdf:type && ?p!=owl:annotatedProperty && ?p!=owl:annotatedTarget && ?p!=owl:annotatedSource)
    }
    """
    )

    for query in queries:
        graph.update(query)

    return graph


def to_rdf_graph(msdf: MappingSetDataFrame) -> Graph:
    """Convert a mapping set dataframe to an RDF graph."""
    doc = to_mapping_set_document(msdf)
    # cntxt = prepare_context(doc.prefix_map)

    # rdflib_dumper.dump(
    #     element=doc.mapping_set,
    #     schemaview=SchemaView(os.path.join(os.getcwd(), "schema/sssom.yaml")),
    #     prefix_map=msdf.prefix_map,
    #     to_file="sssom.ttl",
    # )
    # graph = Graph()
    # graph = graph.parse("sssom.ttl", format="ttl")

    # os.remove("sssom.ttl")  # remove the intermediate file.
    graph = rdflib_dumper.as_rdf_graph(
        element=doc.mapping_set,
        schemaview=SchemaView(SCHEMA_YAML),
        prefix_map=msdf.prefix_map,
    )
    return graph


def to_fhir_json(msdf: MappingSetDataFrame) -> Dict:
    """Convert a mapping set dataframe to a JSON object.

    :param msdf: MappingSetDataFrame: Collection of mappings represented as DataFrame, together w/ additional metadata.
    :return: Dict: A Dictionary serializable as JSON.

    Resources:
      - ConcpetMap::SSSOM mapping spreadsheet: https://docs.google.com/spreadsheets/d/1J19foBAYO8PCHwOfksaIGjNu-q5ILUKFh2HpOCgYle0/edit#gid=1389897118

    TODOs
    todo: when/how to conform to R5 instead of R4?: https://build.fhir.org/conceptmap.html
    TODO: Add additional fields from both specs
     - ConceptMap spec fields: https://www.hl7.org/fhir/r4/conceptmap.html
      - Joe: Can also utilize: /Users/joeflack4/projects/hapi-fhir-jpaserver-starter/_archive/issues/sssom/example_json/minimal.json
     - SSSOM more fields:
     - prefix_map
     - SSSOM spec fields: https://mapping-commons.github.io/sssom/Mapping/
    """
    df: pd.DataFrame = msdf.df
    # Intermediary variables
    metadata: Dict[str, Any] = msdf.metadata if msdf.metadata is not None else {}
    mapping_set_id = metadata.get("mapping_set_id", "")
    name: str = mapping_set_id.split("/")[-1].replace(".sssom.tsv", "")
    # Construct JSON
    # TODO: Fix: sssom/writers.py:293: error: Item "None" of "Optional[Dict[str, Any]]" has no attribute "get"
    #  ...a. Maybe remove the typing? b. remove the get? c. do outside of dict and add after?, d. Add "None"? maybe cant be done here
    #  ...e. Probably assign metadata to new object and use that instead. so won't read as None
    json_obj = {
        "resourceType": "ConceptMap",
        "url": mapping_set_id,
        "identifier": [
            {
                "system": "/".join(mapping_set_id.split("/")[:-1]) + "/",
                "value": mapping_set_id,
            }
        ],
        "version": metadata.get("mapping_set_version", ""),
        "name": name,
        "title": name,
        "status": "draft",  # todo: when done: draft | active | retired | unknown
        "experimental": True,  # todo: False when converter finished
        # todo: should this be date of last converted to FHIR json instead?
        "date": metadata.get("mapping_date", ""),
        # "publisher": "HL7, Inc",  # todo: conceptmap
        # "contact": [{  # todo: conceptmap
        #     "name": "FHIR project team (example)",
        #     "telecom": [{
        #         "system": "url",
        #         "value": "http://hl7.org/fhir"}]
        # }],
        # "description": "",  # todo: conceptmap
        # "useContext": [{  # todo: conceptmap
        #     "code": {
        #         "system": "http://terminology.hl7.org/CodeSystem/usage-context-type",
        #         "code": "venue" },
        #     "valueCodeableConcept": {
        #         "text": "for CCDA Usage" }
        # }],
        # "jurisdiction": [{  # todo: conceptmap
        #     "coding": [{
        #         "system": "urn:iso:std:iso:3166",
        #         "code": "US" }]
        # }],
        # "purpose": "",  # todo: conceptmap
        "copyright": metadata.get("license", ""),
        "sourceUri": metadata.get("subject_source", ""),  # todo: correct?
        "targetUri": metadata.get("object_source", ""),  # todo: correct?
        "group": [
            {
                "source": metadata.get("subject_source", ""),  # todo: correct?
                "target": metadata.get("object_source", ""),  # todo: correct?
                "element": [
                    {
                        "code": row["subject_id"],
                        "display": row.get("subject_label", ""),
                        "target": [
                            {
                                "code": row["object_id"],
                                "display": row.get("object_label", ""),
                                # TODO: R4 (try this first)
                                #  relatedto | equivalent | equal | wider | subsumes | narrower | specializes | inexact | unmatched | disjoint
                                #  https://www.hl7.org/fhir/r4/conceptmap.html
                                # todo: r4: if not found, eventually needs to be `null` or something. check docs to see if nullable, else ask on Zulip
                                # TODO: R5 Needs to be one of:
                                #  related-to | equivalent | source-is-narrower-than-target | source-is-broader-than-target | not-related-to
                                #  https://www.hl7.org/fhir/r4/valueset-concept-map-equivalence.html
                                #  ill update that next time. i can map SSSOM SKOS/etc mappings to FHIR ones
                                #  and then add the original SSSOM mapping CURIE fields somewhere else
                                # https://www.hl7.org/fhir/valueset-concept-map-equivalence.html
                                # https://github.com/mapping-commons/sssom-py/issues/258
                                "equivalence": {
                                    # relateedto: The concepts are related to each other, and have at least some overlap
                                    # in meaning, but the exact relationship is not known.
                                    "skos:related": "relatedto",
                                    "skos:relatedMatch": "relatedto",  # canonical
                                    # equivalent: The definitions of the concepts mean the same thing (including when
                                    # structural implications of meaning are considered) (i.e. extensionally identical).
                                    "skos:exactMatch": "equivalent",
                                    # equal: The definitions of the concepts are exactly the same (i.e. only grammatical
                                    # differences) and structural implications of meaning are identical or irrelevant
                                    # (i.e. intentionally identical).
                                    "equal": "equal",  # todo what's difference between this and above? which to use?
                                    # wider: The target mapping is wider in meaning than the source concept.
                                    "skos:broader": "wider",
                                    "skos:broadMatch": "wider",  # canonical
                                    # subsumes: The target mapping subsumes the meaning of the source concept (e.g. the
                                    # source is-a target).
                                    "rdfs:subClassOf": "subsumes",
                                    "owl:subClassOf": "subsumes",
                                    # narrower: The target mapping is narrower in meaning than the source concept. The
                                    # sense in which the mapping is narrower SHALL be described in the comments in this
                                    # case, and applications should be careful when attempting to use these mappings
                                    # operationally.
                                    "skos:narrower": "narrower",
                                    "skos:narrowMatch": "narrower",  # canonical
                                    # specializes: The target mapping specializes the meaning of the source concept
                                    # (e.g. the target is-a source).
                                    "sssom:superClassOf": "specializes",
                                    # inexact: The target mapping overlaps with the source concept, but both source and
                                    # target cover additional meaning, or the definitions are imprecise and it is
                                    # uncertain whether they have the same boundaries to their meaning. The sense in
                                    # which the mapping is inexact SHALL be described in the comments in this case, and
                                    # applications should be careful when attempting to use these mappings operationally
                                    "skos:closeMatch": "inexact",
                                    # unmatched: There is no match for this concept in the target code system.
                                    # todo: unmatched: this is more complicated. This will be a combination of
                                    #  predicate_id and predicate_modifier (if present). See:
                                    #  https://github.com/mapping-commons/sssom/issues/185
                                    "unmatched": "unmatched",
                                    # disjoint: This is an explicit assertion that there is no mapping between the
                                    # source and target concept.
                                    "owl:disjointWith": "disjoint",
                                }.get(
                                    row["predicate_id"], row["predicate_id"]
                                ),  # r4
                                # "relationship": row['predicate_id'],  # r5
                                # "comment": '',
                                "extension": [
                                    {
                                        # todo: `mapping_justification` consider changing `ValueString` -> `ValueCoding`
                                        #  ...that is, if I happen to know the categories/codes for this categorical variable
                                        #  ...if i do that, do i also need to upload that coding as a (i) `ValueSet` resource? (or (ii) codeable concept? prolly (i))
                                        "url": "http://example.org/fhir/StructureDefinition/mapping_justification",
                                        "ValueString": row.get(
                                            "mapping_justification",
                                            row.get("mapping_justification", ""),
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                    for i, row in df.iterrows()
                ],
                # "unmapped": {  # todo: conceptmap
                #     "mode": "fixed",
                #     "code": "temp",
                #     "display": "temp"
                # }
            }
        ],
    }

    # Delete empty fields
    # todo: This should be recursive?
    keys_to_delete: List[str] = []
    for k, v in json_obj.items():
        if v in [
            None,
            "",
        ]:  # need to allow for `0`, `False`, and maybe some other cases
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del json_obj[k]

    return json_obj


def to_json(msdf: MappingSetDataFrame) -> JsonObj:
    """Convert a mapping set dataframe to a JSON object."""
    doc = to_mapping_set_document(msdf)
    context = prepare_context_str(doc.prefix_map)
    data = JSONDumper().dumps(doc.mapping_set, contexts=context)
    json_obj = json.loads(data)
    return json_obj


def to_ontoportal_json(msdf: MappingSetDataFrame) -> List[Dict]:
    """Convert a mapping set dataframe to a list of ontoportal mapping JSON nbjects."""
    prefix_map = msdf.prefix_map
    metadata: Dict[str, Any] = msdf.metadata if msdf.metadata is not None else {}
    m_list = []

    def resolve(x):
        return _resolve_url(x, prefix_map)

    if msdf.df is not None:
        for _, row in msdf.df.iterrows():
            json_obj = {
                "classes": [resolve(row["subject_id"]), resolve(row["object_id"])],
                "subject_source_id": _resolve_prefix(
                    row.get("subject_source", ""), prefix_map
                ),
                "object_source_id": _resolve_prefix(
                    row.get("object_source", ""), prefix_map
                ),
                "source_name": metadata.get("mapping_set_id", ""),
                "source_contact_info": ",".join(metadata.get("creator_id", "")),
                "date": metadata.get("mapping_date", row.get("mapping_date", "")),
                "name": metadata.get("mapping_set_description", ""),
                "source": resolve(row.get("mapping_justification", "")),
                "comment": row.get("comment", ""),
                "relation": [resolve(row["predicate_id"])],
            }
            m_list.append(json_obj)

    return m_list


# Support methods


def get_writer_function(
    *, output_format: Optional[str] = None, output: TextIO
) -> Tuple[MSDFWriter, str]:
    """Get appropriate writer function based on file format.

    :param output: Output file
    :param output_format: Output file format, defaults to None
    :raises ValueError: Unknown output format
    :return: Type of writer function
    """
    if output_format is None:
        output_format = get_file_extension(output)

    if output_format == "tsv":
        return write_table, output_format
    elif output_format in RDF_FORMATS:
        return write_rdf, output_format
    elif output_format == "rdf":
        return write_rdf, SSSOM_DEFAULT_RDF_SERIALISATION
    elif output_format == "json":
        return write_json, output_format
    elif output_format == "fhir_json":
        return write_fhir_json, output_format
    elif output_format == "ontoportal_json":
        return write_ontoportal_json, output_format
    elif output_format == "owl":
        return write_owl, SSSOM_DEFAULT_RDF_SERIALISATION
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def write_tables(
    sssom_dict: Dict[str, MappingSetDataFrame], output_dir: Union[str, Path]
) -> None:
    """Write table from MappingSetDataFrame object.

    :param sssom_dict: Dictionary of MappingSetDataframes
    :param output_dir: The directory in which the derived SSSOM files are written
    """
    # FIXME documentation does not actually describe what this is doing
    # FIXME explanation of sssom_dict does not make sense
    # FIXME sssom_dict is a bad variable name
    output_dir = Path(output_dir).resolve()
    for split_id, msdf in sssom_dict.items():
        path = output_dir.joinpath(f"{split_id}.sssom.tsv")
        with path.open("w") as file:
            write_table(msdf, file)
        logging.info(f"Writing {path} complete!")


def _inject_annotation_properties(graph: Graph, elements) -> None:
    for var in [
        slot
        for slot in dir(slots)
        if not callable(getattr(slots, slot)) and not slot.startswith("__")
    ]:
        slot = getattr(slots, var)
        if slot.name in elements:
            if slot.uri.startswith(SSSOM_NS):
                graph.add(
                    (
                        URIRef(slot.uri),
                        URIRef(RDF_TYPE),
                        URIRef(OWL_ANNOTATION_PROPERTY),
                    )
                )


def _get_separator(serialisation: Optional[str] = None) -> str:
    if serialisation == "csv":
        sep = ","
    elif serialisation == "tsv" or serialisation is None:
        sep = "\t"
    else:
        raise ValueError(
            f"Unknown table format: {serialisation}, should be one of tsv or csv"
        )
    return sep


def _resolve_url(prefixed_url_str: str, prefix_map: PrefixMap) -> str:
    if not prefixed_url_str:
        return prefixed_url_str

    prefix_url = prefixed_url_str.split(":")
    if len(prefix_url) != 2:
        return prefixed_url_str
    else:
        return _resolve_prefix(prefix_url[0], prefix_map) + prefix_url[1]


def _resolve_prefix(prefix_str, prefix_map: PrefixMap) -> str:
    return prefix_map.get(prefix_str, prefix_str + ":")
