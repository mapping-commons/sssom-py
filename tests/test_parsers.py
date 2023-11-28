"""Tests for parsers."""

import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from xml.dom import minidom

import numpy as np
import pandas as pd
import yaml
from curies import Converter, chain
from rdflib import Graph

from sssom.constants import CURIE_MAP, DEFAULT_LICENSE, SSSOM_URI_PREFIX, get_default_metadata, EXTENDED_PREFIX_MAP
from sssom.context import SSSOM_BUILT_IN_PREFIXES, ensure_converter, get_converter, _get_built_in_prefix_map
from sssom.io import parse_file
from sssom.parsers import (
    _open_input,
    _read_pandas_and_metadata,
    from_alignment_minidom,
    from_obographs,
    from_sssom_dataframe,
    from_sssom_json,
    from_sssom_rdf,
    parse_sssom_table,
)
from sssom.util import MappingSetDataFrame, sort_df_rows_columns
from sssom.writers import write_table
from tests.test_data import data_dir as test_data_dir
from tests.test_data import test_out_dir


class TestParse(unittest.TestCase):
    """A test case for parser functionality."""

    def setUp(self) -> None:
        """Set up the test case."""
        # TODO: change back to the commented url.
        self.df_url = (
            "https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic.tsv"
        )
        self.rdf_graph_file = f"{test_data_dir}/basic.sssom.rdf"
        self.rdf_graph = Graph()
        self.rdf_graph.parse(self.rdf_graph_file, format="ttl")

        self.df_file = f"{test_data_dir}/basic-meta-external.tsv"
        self.df = pd.read_csv(self.df_file, sep="\t", low_memory=False)

        self.obographs_file = f"{test_data_dir}/pato.json"
        with open(self.obographs_file) as json_file:
            self.obographs = json.load(json_file)

        self.broken_obograph_file = f"{test_data_dir}/omim.json"
        with open(self.broken_obograph_file) as json_file:
            self.broken_obographs = json.load(json_file)

        self.json_file = f"{test_data_dir}/basic.json"
        with open(self.json_file) as json_file:
            self.json = json.load(json_file)

        with open(f"{test_data_dir}/basic-meta-external.yml") as file:
            df_meta = yaml.safe_load(file)
            self.df_converter = Converter.from_prefix_map(df_meta.pop(CURIE_MAP))
            self.df_meta = df_meta

        self.alignmentxml_file = f"{test_data_dir}/oaei-ordo-hp.rdf"
        self.alignmentxml = minidom.parse(self.alignmentxml_file)
        self.metadata = get_default_metadata()
        self.converter = get_converter()

    def test_parse_sssom_dataframe_from_file(self):
        """Test parsing a TSV."""
        input_path = f"{test_data_dir}/basic.tsv"
        msdf = parse_sssom_table(input_path)
        output_path = os.path.join(test_out_dir, "test_parse_sssom_dataframe.tsv")
        with open(output_path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            141,
            f"{input_path} has the wrong number of mappings.",
        )

    def test_parse_sssom_dataframe_from_stringio(self):
        """Test parsing a TSV."""
        input_path = f"{test_data_dir}/basic.tsv"
        with open(input_path, "r") as file:
            input_string = file.read()
        stream = io.StringIO(input_string)
        msdf = parse_sssom_table(stream)
        output_path = os.path.join(test_out_dir, "test_parse_sssom_dataframe_stream.tsv")
        with open(output_path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            141,
            f"{input_path} has the wrong number of mappings.",
        )

    def test_parse_sssom_dataframe_from_url(self):
        """Test parsing a TSV from a URL."""
        msdf = parse_sssom_table(self.df_url)
        output_path = os.path.join(test_out_dir, "test_parse_sssom_dataframe_url.tsv")
        with open(output_path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.df_url} has the wrong number of mappings.",
        )

    def test_parse_obographs(self):
        """Test parsing OBO Graph JSON."""
        msdf = from_obographs(
            jsondoc=self.obographs,
            prefix_map=self.converter,
            meta=self.metadata,
        )
        path = os.path.join(test_out_dir, "test_parse_obographs.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            # this number went up from 8099 when the curies.Converter was introduced
            # since it was able to handle CURIE prefix and URI prefix synonyms
            8489,
            len(msdf.df),
            f"{self.obographs_file} has the wrong number of mappings.",
        )

    def test_parse_tsv(self):
        """Test parsing TSV."""
        msdf = from_sssom_dataframe(df=self.df, prefix_map=self.df_converter, meta=self.df_meta)
        path = os.path.join(test_out_dir, "test_parse_tsv.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.df_file} has the wrong number of mappings.",
        )

    def test_parse_alignment_minidom(self):
        """Test parsing an alignment XML."""
        msdf = from_alignment_minidom(
            dom=self.alignmentxml,
            prefix_map=self.converter,
            meta=self.metadata,
        )
        path = os.path.join(test_out_dir, "test_parse_alignment_minidom.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            646,
            f"{self.alignmentxml_file} has the wrong number of mappings.",
        )

    def test_parse_alignment_xml(self):
        """Test parsing an alignment XML.

        This issue should fail because entity 1 of the second mapping
        is not in prefix map.
        """
        alignment_api_xml = dedent(
            """\
            <?xml version="1.0" encoding="utf-8"?>
            <rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
                xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
                <Alignment>
                    <xml>yes</xml>
                    <level>0</level>
                    <type>??</type>
                    <onto1>http://purl.obolibrary.org/obo/fbbt.owl</onto1>
                    <onto2>http://purl.obolibrary.org/obo/wbbt.owl</onto2>
                    <uri1>http://purl.obolibrary.org/obo/fbbt.owl</uri1>
                    <uri2>http://purl.obolibrary.org/obo/wbbt.owl</uri2>
                    <map>
                        <Cell>
                            <entity1 rdf:resource="http://purl.obolibrary.org/obo/FBbt_00004924"/>
                            <entity2 rdf:resource="http://purl.obolibrary.org/obo/WBbt_0006760"/>
                            <measure rdf:datatype="xsd:float">0.75</measure>
                            <relation>=</relation>
                        </Cell>
                    </map>
                    <map>
                        <Cell>"
                            <entity1 rdf:resource="http://randomurlwithnochancetobeinprefixmap.org/ID_123"/>
                            <entity2 rdf:resource="http://purl.obolibrary.org/obo/WBbt_0005815"/>
                            <measure rdf:datatype="xsd:float">0.5</measure>
                            <relation>=</relation>
                        </Cell>
                    </map>
                </Alignment>
            </rdf:RDF>
            """
        )
        alignmentxml = minidom.parseString(alignment_api_xml)

        prefix_map_without_prefix = {
            "WBbt": "http://purl.obolibrary.org/obo/WBbt_",
            "FBbt": "http://purl.obolibrary.org/obo/FBbt_",
        }

        prefix_map_with_prefix = {
            "WBbt": "http://purl.obolibrary.org/obo/WBbt_",
            "FBbt": "http://purl.obolibrary.org/obo/FBbt_",
            "ID": "http://randomurlwithnochancetobeinprefixmap.org/ID_",
        }

        msdf_with_broken_prefixmap = from_alignment_minidom(
            dom=alignmentxml,
            prefix_map=prefix_map_without_prefix,
        )
        expected_row_values = [
            "FBbt:00004924",
            "skos:exactMatch",
            "WBbt:0006760",
            "semapv:UnspecifiedMatching",
            0.75,
        ]
        self.assertEqual(expected_row_values, msdf_with_broken_prefixmap.df.iloc[0].tolist())

        msdf_with_prefixmap = from_alignment_minidom(
            dom=alignmentxml,
            prefix_map=prefix_map_with_prefix,
        )
        expected_row_values2 = [
            "ID:123",
            "skos:exactMatch",
            "WBbt:0005815",
            "semapv:UnspecifiedMatching",
            0.5,
        ]
        self.assertEqual(expected_row_values, msdf_with_prefixmap.df.iloc[0].tolist())
        self.assertEqual(expected_row_values2, msdf_with_prefixmap.df.iloc[1].tolist())

        msdf_without_prefixmap = from_alignment_minidom(
            dom=alignmentxml,
        )
        self.assertEqual(expected_row_values, msdf_without_prefixmap.df.iloc[0].tolist())

    def test_parse_sssom_rdf(self):
        """Test parsing RDF."""
        msdf = from_sssom_rdf(g=self.rdf_graph, prefix_map=self.df_converter, meta=self.metadata)
        path = os.path.join(test_out_dir, "test_parse_sssom_rdf.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            136,
            f"{self.rdf_graph_file} has the wrong number of mappings.",
        )

    def test_parse_sssom_json(self):
        """Test parsing JSON."""
        msdf = from_sssom_json(
            jsondoc=self.json,
            prefix_map=self.df_converter,
            meta=self.metadata,
        )
        path = os.path.join(test_out_dir, "test_parse_sssom_json.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.json_file} has the wrong number of mappings.",
        )

    # * "mapping_justification" is no longer multivalued.
    # def test_piped_element_to_list(self):
    #     """Test for multi-valued element (piped in SSSOM tables) to list."""
    #     input_path = os.path.join(test_data_dir, "basic.tsv")
    #     msdf = parse_sssom_table(input_path)
    #     df = msdf.df
    #     msdf.df = df[
    #         df["mapping_justification"].str.contains("\\|", na=False)
    #     ].reset_index()
    #     old_match_type = msdf.df["mapping_justification"]
    #     msdoc = to_mapping_set_document(msdf)
    #     new_msdf = to_mapping_set_dataframe(msdoc)
    #     new_match_type = new_msdf.df["mapping_justification"]
    #     self.assertTrue(old_match_type.equals(new_match_type))

    def test_read_sssom_table(self):
        """Test read SSSOM method to validate import of all columns."""
        input_path = os.path.join(test_data_dir, "basic3.tsv")
        msdf = parse_sssom_table(input_path)
        imported_df = pd.read_csv(input_path, comment="#", sep="\t").fillna("")
        imported_df = sort_df_rows_columns(imported_df)
        msdf.df = sort_df_rows_columns(msdf.df)
        self.assertEqual(set(imported_df.columns), set(msdf.df.columns))
        list_cols = [
            "subject_match_field",
            "object_match_field",
            "match_string",
            "mapping_justification",
        ]
        for idx, row in msdf.df.iterrows():
            for k, v in row.items():
                if v == np.nan:
                    self.assertTrue(math.isnan(imported_df.iloc[idx][k]))
                else:
                    if k not in list_cols:
                        if v is np.nan:
                            self.assertTrue(imported_df.iloc[idx][k] is v)
                        else:
                            self.assertEqual(imported_df.iloc[idx][k], v)
                    elif k == "mapping_justification":
                        self.assertEqual(imported_df.iloc[idx][k], v)
                    else:
                        self.assertEqual(imported_df.iloc[idx][k], v)

    def test_parse_obographs_merged(self):
        """Test parsing OBO Graph JSON using custom prefix_map."""
        hp_json = f"{test_data_dir}/hp-subset.json"
        hp_meta = f"{test_data_dir}/hp-subset-metadata.yml"
        outfile = f"{test_out_dir}/hp-subset-parse.tsv"

        with open(hp_meta, "r") as f:
            data = yaml.safe_load(f)
            custom_curie_map = data["curie_map"]

        with open(outfile, "w") as f:
            parse_file(
                input_path=hp_json,
                prefix_map_mode="merged",
                clean_prefixes=True,
                input_format="obographs-json",
                metadata_path=hp_meta,
                output=f,
            )
        msdf = parse_sssom_table(outfile)
        self.assertTrue(custom_curie_map.items() <= msdf.prefix_map.items())


class TestParseExplicit(unittest.TestCase):
    """This test case contains explicit tests for parsing."""

    def test_round_trip(self):
        """Explicitly test round tripping."""
        rows = [
            (
                "DOID:0050601",
                "ADULT syndrome",
                "skos:exactMatch",
                "UMLS:C1863204",
                "ADULT SYNDROME",
                "semapv:ManualMappingCuration",
                "orcid:0000-0003-4423-4370",
            )
        ]
        columns = [
            "subject_id",
            "subject_label",
            "predicate_id",
            "object_id",
            "object_label",
            "mapping_justification",
            "creator_id",
        ]
        df = pd.DataFrame(rows, columns=columns)
        msdf = MappingSetDataFrame(df=df, converter=ensure_converter())
        msdf.clean_prefix_map(strict=True)
        #: This is a set of the prefixes that explicitly are used in this
        #: example. SSSOM-py also adds the remaining builtin prefixes from
        #: :data:`sssom.context.SSSOM_BUILT_IN_PREFIXES`, which is reflected
        #: in the formulation of the test expectation below
        explicit_prefixes = {"DOID", "semapv", "orcid", "skos", "UMLS"}
        self.assertEqual(
            explicit_prefixes.union(SSSOM_BUILT_IN_PREFIXES),
            set(msdf.prefix_map),
        )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory.joinpath("test.sssom.tsv")
            with path.open("w") as file:
                write_table(msdf, file)

            _, read_metadata = _read_pandas_and_metadata(_open_input(path))
            reconsitited_msdf = parse_sssom_table(path)

        # This tests what's actually in the file after it's written out
        self.assertEqual({CURIE_MAP, "license", "mapping_set_id"}, set(read_metadata))
        self.assertEqual(DEFAULT_LICENSE, read_metadata["license"])
        self.assertTrue(read_metadata["mapping_set_id"].startswith(f"{SSSOM_URI_PREFIX}mappings/"))

        expected_prefix_map = {
            "DOID": "http://purl.obolibrary.org/obo/DOID_",
            "UMLS": "http://linkedlifedata.com/resource/umls/id/",
            "orcid": "https://orcid.org/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "semapv": "https://w3id.org/semapv/vocab/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "sssom": "https://w3id.org/sssom/",
        }
        self.assertEqual(
            expected_prefix_map,
            read_metadata[CURIE_MAP],
        )

        # This checks that nothing funny gets added unexpectedly
        self.assertEqual(expected_prefix_map, reconsitited_msdf.prefix_map)

    def test_bimap(self):
        epm = [{
        "prefix": "Orphanet",
        "prefix_synonyms": [
            "orphanet.ordo"
        ],
        "uri_prefix": "http://www.orpha.net/ORDO/Orphanet_" }]
        converter = Converter.from_extended_prefix_map(epm)
        self.assertTrue('Orphanet' in converter.prefix_map)
        compressed = converter.compress("http://www.orpha.net/ORDO/Orphanet_123")
        self.assertEqual(compressed,"Orphanet:123")

    def test_bimap2(self):
        """Explicitly test round tripping."""

        # json_string = """{
        #   "graphs" : [ {
        #     "id" : "http://purl.obolibrary.org/obo/mondo.owl",
        #     "meta" : {
        #       "version" : "http://purl.obolibrary.org/obo/mondo/releases/2023-09-12/mondo.owl"
        #     },
        #     "nodes" : [ {
        #       "id" : "http://purl.obolibrary.org/obo/MONDO_0009650",
        #       "lbl" : "Some label",
        #       "type" : "CLASS",
        #       "meta" : {
        #         "definition" : {
        #           "val" : "Some def"
        #         },
        #         "xrefs" : [ {
        #           "val" : "Orphanet:576"
        #         }],
        #         "basicPropertyValues" : [ {
        #           "pred" : "http://www.w3.org/2004/02/skos/core#exactMatch",
        #           "val" : "http://www.orpha.net/ORDO/Orphanet_576"
        #      } ] }
        #   } ]
        #   } ]
        # }"""
        # input = json.loads(json_string)
        json_file = f"{test_data_dir}/mirror-mondo.json"
        with open(json_file) as f:
            input = json.load(f)

        msdf = from_obographs(
            jsondoc=input,
        )
        msdf.clean_prefix_map(strict=True)

        # ordo_id = msdf.df['object_id'].iloc[0]
        # self.assertTrue(ordo_id.startswith("Orphanet:"))
        self.assertNotIn("ordo.orphanet", msdf.prefix_map)
