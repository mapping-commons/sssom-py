"""Serialization functions for SSSOM."""

import json
import logging as _logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import pandas as pd
import yaml
from curies import Converter
from deprecation import deprecated
from jsonasobj2 import JsonObj
from linkml_runtime.dumpers import JSONDumper, rdflib_dumper
from linkml_runtime.utils.schemaview import SchemaView
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF
from sssom_schema import slots

from sssom.validators import check_all_prefixes_in_curie_map

from .constants import CURIE_MAP, SCHEMA_YAML, SSSOM_URI_PREFIX
from .context import _load_sssom_context
from .parsers import to_mapping_set_document
from .util import (
    RDF_FORMATS,
    SSSOM_DEFAULT_RDF_SERIALISATION,
    URI_SSSOM_MAPPINGS,
    MappingSetDataFrame,
    get_file_extension,
    invert_mappings,
    sort_df_rows_columns,
)

logging = _logging.getLogger(__name__)

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
    sep = _get_separator(serialisation)

    meta: Dict[str, Any] = {}
    meta.update(msdf.metadata)
    meta[CURIE_MAP] = msdf.converter.bimap
    if sort:
        msdf.df = sort_df_rows_columns(msdf.df)

    if embedded_mode:
        lines = yaml.safe_dump(meta).split("\n")
        lines = [f"# {line}" for line in lines if line != ""]
        s = msdf.df.to_csv(sep=sep, index=False).rstrip("\n")
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


def write_json(msdf: MappingSetDataFrame, output: TextIO, serialisation="json") -> None:
    """Write a mapping set dataframe to the file as JSON.

    :param serialisation: The JSON format to use. Supported formats are:
     - fhir_json: Outputs JSON in FHIR ConceptMap format (https://fhir-ru.github.io/conceptmap.html)
       https://mapping-commons.github.io/sssom-py/sssom.html#sssom.writers.to_fhir_json
     - json: Outputs to SSSOM JSON https://mapping-commons.github.io/sssom-py/sssom.html#sssom.writers.to_json
     - ontoportal_json: Outputs JSON in Ontoportal format (https://ontoportal.org/)
       https://mapping-commons.github.io/sssom-py/sssom.html#sssom.writers.to_ontoportal_json
    """
    func_map: Dict[str, Callable] = {
        "fhir_json": to_fhir_json,
        "json": to_json,
        "ontoportal_json": to_ontoportal_json,
    }
    if serialisation not in func_map:
        raise ValueError(
            f"Unknown JSON format: {serialisation}. Supported flavors: {', '.join(func_map.keys())}"
        )
    func: Callable = func_map[serialisation]
    data = func(msdf)
    json.dump(data, output, indent=2)


@deprecated(deprecated_in="0.4.7", details="Use write_json() instead")
def write_fhir_json(msdf: MappingSetDataFrame, output: TextIO, serialisation="fhir_json") -> None:
    """Write a mapping set dataframe to the file as FHIR ConceptMap JSON."""
    if serialisation != "fhir_json":
        raise ValueError(
            f"Unknown json format: {serialisation}, currently only fhir_json supported"
        )
    write_json(msdf, output, serialisation="fhir_json")


@deprecated(deprecated_in="0.4.7", details="Use write_json() instead")
def write_ontoportal_json(
    msdf: MappingSetDataFrame, output: TextIO, serialisation: str = "ontoportal_json"
) -> None:
    """Write a mapping set dataframe to the file as the ontoportal mapping JSON model."""
    if serialisation != "ontoportal_json":
        raise ValueError(
            f"Unknown json format: {serialisation}, currently only ontoportal_json supported"
        )
    write_json(msdf, output, serialisation="ontoportal_json")


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


# Converters
# Converters convert a mappingsetdataframe to an object of the supportes types (json, pandas dataframe)


def to_owl_graph(msdf: MappingSetDataFrame) -> Graph:
    """Convert a mapping set dataframe to OWL in an RDF graph."""
    msdf.df = invert_mappings(
        df=msdf.df,
        merge_inverted=False,
        update_justification=False,
        predicate_invert_dictionary={"sssom:superClassOf": "rdfs:subClassOf"},
    )
    graph = to_rdf_graph(msdf=msdf)

    for _s, _p, o in graph.triples((None, URIRef(URI_SSSOM_MAPPINGS), None)):
        graph.add((o, URIRef(RDF_TYPE), OWL.Axiom))

    for axiom in graph.subjects(RDF.type, OWL.Axiom):
        for p in graph.objects(subject=axiom, predicate=OWL.annotatedProperty):
            for s in graph.objects(subject=axiom, predicate=OWL.annotatedSource):
                for o in graph.objects(subject=axiom, predicate=OWL.annotatedTarget):
                    graph.add((s, p, o))

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
    graph = rdflib_dumper.as_rdf_graph(
        element=doc.mapping_set,
        schemaview=SchemaView(SCHEMA_YAML),
        # TODO Use msdf.converter directly via https://github.com/linkml/linkml-runtime/pull/278
        prefix_map=msdf.converter.bimap,
    )
    return graph


def to_fhir_json(msdf: MappingSetDataFrame) -> Dict:
    """Convert a mapping set dataframe to a JSON object.

    :param msdf: MappingSetDataFrame: Collection of mappings represented as DataFrame, together w/ additional metadata.
    :return: Dict: A Dictionary serializable as JSON.

    Resources:
      - ConceptMap::SSSOM mapping spreadsheet:
      https://docs.google.com/spreadsheets/d/1J19foBAYO8PCHwOfksaIGjNu-q5ILUKFh2HpOCgYle0/edit#gid=1389897118

    TODO: add to CLI & to these functions: r4 vs r5 param
    TODO: What if the msdf doesn't have everything we need? (i) metadata, e.g. yml, (ii) what if we need to override?
     - todo: later: allow any nested arbitrary override: (get in kwargs, else metadata.get(key, None))

    Minor todos
    todo: mapping_justification: consider `ValueString` -> `ValueCoding` https://github.com/timsbiomed/issues/issues/152
    todo: when/how to conform to R5 instead of R4?: https://build.fhir.org/conceptmap.html
    """
    # Constants
    df: pd.DataFrame = msdf.df
    # TODO: R4 (try this first)
    #  relatedto | equivalent | equal | wider | subsumes | narrower | specializes | inexact | unmatched | disjoint
    #  https://www.hl7.org/fhir/r4/conceptmap.html
    # todo: r4: if not found, should likely be `null` or something. check docs to see if nullable, else ask on Zulip
    # TODO: R5 Needs to be one of:
    #  related-to | equivalent | source-is-narrower-than-target | source-is-broader-than-target | not-related-to
    #  https://www.hl7.org/fhir/r4/valueset-concept-map-equivalence.html
    #  ill update that next time. i can map SSSOM SKOS/etc mappings to FHIR ones
    #  and then add the original SSSOM mapping CURIE fields somewhere else
    # https://www.hl7.org/fhir/valueset-concept-map-equivalence.html
    # https://github.com/mapping-commons/sssom-py/issues/258
    equivalence_map = {
        # relateedto: The concepts are related to each other, and have at least some overlap in meaning, but the exact
        #   relationship is not known.
        "skos:related": "relatedto",
        "skos:relatedMatch": "relatedto",  # canonical
        # equivalent: The definitions of the concepts mean the same thing (including when structural implications of
        #   meaning are considered) (i.e. extensionally identical).
        "skos:exactMatch": "equivalent",
        # equal: The definitions of the concepts are exactly the same (i.e. only grammatical differences) and structural
        #   implications of meaning are identical or irrelevant (i.e. intentionally identical).
        "equal": "equal",  # todo what's difference between this and above? which to use?
        # wider: The target mapping is wider in meaning than the source concept.
        "skos:broader": "wider",
        "skos:broadMatch": "wider",  # canonical
        # subsumes: The target mapping subsumes the meaning of the source concept (e.g. the source is-a target).
        "rdfs:subClassOf": "subsumes",
        # narrower: The target mapping is narrower in meaning than the source concept. The sense in which the mapping is
        #   narrower SHALL be described in the comments in this case, and applications should be careful when attempting
        #   to use these mappings operationally.
        "skos:narrower": "narrower",
        "skos:narrowMatch": "narrower",  # canonical
        # specializes: The target mapping specializes the meaning of the source concept (e.g. the target is-a source).
        "sssom:superClassOf": "specializes",
        # inexact: The target mapping overlaps with the source concept, but both source and target cover additional
        # meaning, or the definitions are imprecise and it is uncertain whether they have the same boundaries to their
        # meaning. The sense in which the mapping is inexact SHALL be described in the comments in this case, and
        # applications should be careful when attempting to use these mappings operationally
        "skos:closeMatch": "inexact",
        # unmatched: There is no match for this concept in the target code system.
        #   todo: this is more complicated. This will be a combination of predicate_id and predicate_modifier (if
        #    present). See: https://github.com/mapping-commons/sssom/issues/185
        "unmatched": "unmatched",
        # disjoint: This is an explicit assertion that there is no mapping between the
        # source and target concept.
        "owl:disjointWith": "disjoint",
    }

    # Intermediary variables
    metadata: Dict[str, Any] = msdf.metadata
    mapping_set_id = metadata.get("mapping_set_id", "")
    name: str = mapping_set_id.split("/")[-1].replace(".sssom.tsv", "")

    # Construct JSON
    json_obj: Dict[str, Any] = {
        "resourceType": "ConceptMap",
        "url": mapping_set_id,
        # Assumes mapping_set_id is a URI w/ artefact name at end. System becomes URI stem, value becomes artefact name
        "identifier": [
            {
                "system": "/".join(mapping_set_id.split("/")[:-1]) + "/",
                "value": mapping_set_id,
            }
        ],
        "version": metadata.get("mapping_set_version", ""),
        "name": name,
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
        # TODO: Might want to make each "group" first, if there is more than 1 set of ontology1::ontology2
        #  ...within a given MappingSet / set of SSSOM TSV rows.
        "group": [
            {
                "element": []
                # "unmapped": {  # todo: conceptmap
                #     "mode": "fixed",
                #     "code": "temp",
                #     "display": "temp"
                # }
            }
        ],
    }
    if "mapping_set_title" in metadata:
        json_obj["title"] = metadata["mapping_set_title"]

    # todo: Override? but how? (2024/04/05 Joe: idr what I was trying to override)
    if "subject_source" in metadata:
        json_obj["sourceUri"] = metadata["subject_source"]
        json_obj["group"][0]["source"] = metadata["subject_source"]
    if "object_source" in metadata:
        json_obj["targetUri"] = metadata["object_source"]
        json_obj["group"][0]["target"] = metadata["object_source"]

    for _i, row in df.iterrows():
        entry = {
            "code": row["subject_id"],
            "display": row.get("subject_label", ""),  # todo: if empty, don't add this key
            "target": [
                {
                    "code": row["object_id"],
                    "display": row.get("object_label", ""),  # todo: if empty, don't add this key
                    "equivalence": equivalence_map.get(
                        row["predicate_id"], row["predicate_id"]
                    ),  # r4
                    # "relationship": row['predicate_id'],  # r5
                    # "comment": '',
                    "extension": [
                        {
                            "url": "http://example.org/fhir/StructureDefinition/mapping_justification",
                            "valueString": row.get(
                                "mapping_justification",
                                row.get(
                                    "mapping_justification", ""
                                ),  # todo: if empty, don't add this key
                            ),
                        }
                    ],
                }
            ],
        }
        json_obj["group"][0]["element"].append(entry)

    # Delete empty fields
    # todo: This should be recursive? yes
    #  - it catches empty 'sourceUri' and 'targetUri', but not 'source' and 'target'
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


def _update_sssom_context_with_prefixmap(converter: Converter):
    """Prepare a JSON-LD context and dump to a string."""
    context = _load_sssom_context()
    for k, v in converter.bimap.items():
        if k in context["@context"] and context["@context"][k] != v:
            logging.info(
                f"{k} namespace is already in the context, ({context['@context'][k]}, "
                f"but with a different value than {v}. Overwriting!"
            )
        context["@context"][k] = v
    return context


def to_json(msdf: MappingSetDataFrame) -> JsonObj:
    """Convert a mapping set dataframe to a JSON object."""
    doc = to_mapping_set_document(msdf)
    context = _update_sssom_context_with_prefixmap(doc.converter)
    data = JSONDumper().dumps(doc.mapping_set, contexts=json.dumps(context))
    json_obj = json.loads(data)
    return json_obj


def to_ontoportal_json(msdf: MappingSetDataFrame) -> List[Dict]:
    """Convert a mapping set dataframe to a list of ontoportal mapping JSON objects."""
    converter = msdf.converter
    metadata: Dict[str, Any] = msdf.metadata
    m_list = []
    for _, row in msdf.df.iterrows():
        mapping_justification = row.get("mapping_justification", "")
        if "creator_id" in row:
            creators = row["creator_id"]
        elif "creator_id" in metadata:
            creators = metadata["creator_id"]
        else:
            creators = []

        json_obj = {
            "classes": [
                converter.expand(row["subject_id"]),
                converter.expand(row["object_id"]),
            ],
            "subject_source_id": row.get("subject_source", ""),
            "object_source_id": row.get("object_source", ""),
            "source_name": metadata.get("mapping_set_id", ""),
            "source_contact_info": ",".join(creators),
            "date": metadata.get("mapping_date", row.get("mapping_date", "")),
            "name": metadata.get("mapping_set_title", ""),
            "source": converter.expand(mapping_justification) if mapping_justification else "",
            "comment": row.get("comment", ""),
            "relation": [converter.expand(row["predicate_id"])],
        }
        json_obj = {k: v for k, v in json_obj.items() if k and v}
        m_list.append(json_obj)
    return m_list


# Support methods

WRITER_FUNCTIONS: Dict[str, Tuple[Callable, Optional[str]]] = {
    "tsv": (write_table, None),
    "owl": (write_owl, SSSOM_DEFAULT_RDF_SERIALISATION),
    "ontoportal_json": (write_json, "ontoportal_json"),
    "fhir_json": (write_json, "fhir_json"),
    "json": (write_json, "json"),
    "rdf": (write_rdf, SSSOM_DEFAULT_RDF_SERIALISATION),
}
for rdf_format in RDF_FORMATS:
    WRITER_FUNCTIONS[rdf_format] = write_rdf, rdf_format


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
    if output_format not in WRITER_FUNCTIONS:
        raise ValueError(f"Unknown output format: {output_format}")
    func, tag = WRITER_FUNCTIONS[output_format]
    return func, tag or output_format


def write_tables(sssom_dict: Dict[str, MappingSetDataFrame], output_dir: Union[str, Path]) -> None:
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
        raise ValueError(f"Unknown table format: {serialisation}, should be one of tsv or csv")
    return sep
