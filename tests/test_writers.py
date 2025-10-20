"""Tests for SSSOM writers."""

import json
import os
import unittest
from typing import Any, Dict

import curies
import pandas as pd
from curies import Converter
from fastapi.testclient import TestClient

from sssom import MappingSetDataFrame
from sssom.constants import (
    CREATOR_ID,
    MAPPING_JUSTIFICATION,
    OBJECT_ID,
    OBJECT_LABEL,
    PREDICATE_ID,
    SEMAPV,
    SUBJECT_ID,
    SUBJECT_LABEL,
)
from sssom.parsers import parse_sssom_json, parse_sssom_rdf, parse_sssom_table
from sssom.writers import (
    EXAMPLE_SPARQL_QUERY,
    _update_sssom_context_with_prefixmap,
    get_rdflib_endpoint_app,
    to_json,
    write_json,
    write_owl,
    write_rdf,
    write_table,
)
from tests.constants import data_dir as test_data_dir
from tests.constants import test_out_dir


class TestWrite(unittest.TestCase):
    """A test case for SSSOM writers."""

    def setUp(self) -> None:
        """Set up the test case with a basic SSSOM example."""
        self.msdf = parse_sssom_table(f"{test_data_dir}/basic.tsv")
        # self.msdf = read_sssom_table(f"{test_data_dir}/basic-simple.tsv")
        self.mapping_count = 141  # 141 for basic.tsv

    def test_write_sssom_dataframe(self) -> None:
        """Test writing as a dataframe."""
        tmp_path = os.path.join(test_out_dir, "test_write_sssom_dataframe.tsv")
        write_table(self.msdf, tmp_path)
        msdf = parse_sssom_table(tmp_path)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{tmp_path} has the wrong number of mappings.",
        )

    def test_write_sssom_rdf(self) -> None:
        """Test writing as RDF."""
        path_1 = os.path.join(test_out_dir, "test_write_sssom_rdf.rdf")
        write_rdf(self.msdf, path_1)
        msdf = parse_sssom_rdf(path_1, self.msdf.prefix_map)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{path_1} has the wrong number of mappings.",
        )

        # TODO this test doesn't make sense
        path_2 = os.path.join(test_out_dir, "test_write_sssom_rdf.rdf.tsv")
        write_table(self.msdf, path_2)

    def test_write_sssom_json(self) -> None:
        """Test writing as JSON."""
        path = os.path.join(test_out_dir, "test_write_sssom_json.json")
        write_json(self.msdf, path)
        msdf = parse_sssom_json(path)
        self.assertEqual(
            len(msdf.df),
            self.mapping_count,
            f"{path} has the wrong number of mappings.",
        )

    def test_write_sssom_json_context(self) -> None:
        """Test when writing to JSON, the context is correctly written as well."""
        rows = [
            (
                "DOID:0050601",
                "ADULT syndrome",
                "skos:exactMatch",
                "UMLS:C1863204",
                "ADULT SYNDROME",
                SEMAPV.ManualMappingCuration.value,
                "orcid:0000-0003-4423-4370",
            )
        ]
        columns = [
            SUBJECT_ID,
            SUBJECT_LABEL,
            PREDICATE_ID,
            OBJECT_ID,
            OBJECT_LABEL,
            SEMAPV.ManualMappingCuration.value,
            CREATOR_ID,
        ]
        df = pd.DataFrame(rows, columns=columns)
        msdf = MappingSetDataFrame(df)
        msdf.clean_prefix_map()
        json_object = to_json(msdf)
        self.assertIn("@context", json_object)
        self.assertIn("DOID", json_object["@context"])
        self.assertIn("mapping_set_id", json_object["@context"])

    def test_update_sssom_context_with_prefixmap(self) -> None:
        """Test when writing to JSON, the context is correctly written as well."""
        records = [
            {
                "prefix": "SCTID",
                "prefix_synonyms": ["snomed"],
                "uri_prefix": "http://snomed.info/id/",
            },
        ]
        converter = Converter.from_extended_prefix_map(records)
        context = _update_sssom_context_with_prefixmap(converter)
        self.assertIn("@context", context)
        self.assertIn("SCTID", context["@context"])
        self.assertNotIn("snomed", context["@context"])
        self.assertIn("mapping_set_id", context["@context"])

    def test_write_sssom_fhir(self) -> None:
        """Test writing as FHIR ConceptMap JSON."""
        # Vars
        path = os.path.join(test_out_dir, "test_write_sssom_fhir.json")
        msdf: MappingSetDataFrame = self.msdf
        metadata: Dict[str, Any] = msdf.metadata
        mapping_set_id: str = metadata["mapping_set_id"]

        # Write
        write_json(self.msdf, path, "fhir_json")
        # Read
        # todo: after implementing reader/importer, change this to `msdf = parse_sssom_fhir_json()`
        with open(path, "r") as file:
            d = json.load(file)
        # Test
        # - metadata
        self.assertEqual(d["resourceType"], "ConceptMap")
        self.assertIn(d["identifier"][0]["system"], mapping_set_id)
        self.assertEqual(len(d["identifier"]), 1)
        self.assertEqual(
            len({d["identifier"][0]["value"], mapping_set_id, d["url"]}), 1
        )  # assert all same
        # todo: if/when more test cases, shan't be just 'basic.tsv'
        self.assertEqual(d["name"], "basic.tsv")
        # self.assertEqual(d["title"], "todo")  # missing from basic.tsv
        self.assertEqual(d["status"], "draft")
        self.assertEqual(d["experimental"], True)
        self.assertEqual(len(d["date"]), len("YYYY-MM-DD"))
        self.assertEqual(d["copyright"], "https://creativecommons.org/publicdomain/zero/1.0/")
        # - n mappings
        self.assertEqual(
            len(d["group"][0]["element"]),
            self.mapping_count,
            f"{path} has the wrong number of mappings.",
        )
        # - more
        self.assertEqual(len(d["group"]), 1)
        # todo: code
        # todo: display
        # todo: equivalence
        #  - I'm getting: subsumes, owl:equivalentClass (and see what else in basic.tsv)
        # todo: mapping_justification extensionprint()  # TODO: temp

    def test_write_sssom_owl(self) -> None:
        """Test writing as OWL."""
        tmp_file = os.path.join(test_out_dir, "test_write_sssom_owl.owl")
        write_owl(self.msdf, tmp_file)

    def test_write_sssom_ontoportal_json(self) -> None:
        """Test writing as ontoportal JSON."""
        path = os.path.join(test_out_dir, "test_write_sssom_ontoportal_json.json")
        write_json(self.msdf, path, "ontoportal_json")

        with open(path, "r") as file:
            d = json.load(file)

        self.assertEqual(
            len(d),
            self.mapping_count,
            f"{path} has the wrong number of mappings.",
        )

    def test_rdflib_endpoint(self) -> None:
        """Test serving SPARQL."""
        rows = [
            (
                "DOID:0050601",
                "skos:exactMatch",
                "UMLS:C1863204",
                SEMAPV.ManualMappingCuration.value,
            )
        ]
        columns = [
            SUBJECT_ID,
            PREDICATE_ID,
            OBJECT_ID,
            MAPPING_JUSTIFICATION,
        ]
        df = pd.DataFrame(rows, columns=columns)
        converter = curies.Converter.from_prefix_map(
            {
                "DOID": "http://purl.obolibrary.org/obo/DOID_",
                "UMLS": "https://uts.nlm.nih.gov/uts/umls/concept/",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "semapv": "https://w3id.org/semapv/vocab/",
            }
        )
        msdf = MappingSetDataFrame(df, converter=converter)
        app = get_rdflib_endpoint_app(msdf)
        client = TestClient(app)
        response = client.get(
            "/", params={"query": EXAMPLE_SPARQL_QUERY}, headers={"Accept": "application/json"}
        )
        self.assertEqual(200, response.status_code)
        results = response.json()["results"]["bindings"]
        self.assertEqual(1, len(results))
        self.assertEqual("http://purl.obolibrary.org/obo/DOID_0050601", results[0]["s"]["value"])
        self.assertEqual(
            "https://uts.nlm.nih.gov/uts/umls/concept/C1863204", results[0]["o"]["value"]
        )
