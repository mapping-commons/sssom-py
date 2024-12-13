"""Tests for parsers."""

import io
import json
import math
import os
import unittest
from collections import ChainMap
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from xml.dom import minidom

import numpy as np
import pandas as pd
import yaml
from curies import Converter
from rdflib import Graph

from sssom.constants import CURIE_MAP, get_default_metadata
from sssom.context import SSSOM_BUILT_IN_PREFIXES, ensure_converter, get_converter
from sssom.io import parse_file
from sssom.parsers import (
    PARSING_FUNCTIONS,
    from_alignment_minidom,
    from_obographs,
    from_sssom_dataframe,
    from_sssom_json,
    from_sssom_rdf,
    parse_sssom_table,
)
from sssom.util import MappingSetDataFrame, sort_df_rows_columns
from sssom.writers import WRITER_FUNCTIONS, write_table
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
            141,
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

    def test_parse_trailing_tabs_in_metadata_header(self):
        """Test parsing a file containing trailing tabs in header."""
        input_path = f"{test_data_dir}/trailing-tabs.sssom.tsv"
        msdf = parse_sssom_table(input_path)
        self.assertEqual(msdf.metadata["mapping_set_id"], "https://example.org/sets/exo2c")
        self.assertEqual(
            len(msdf.df),
            8,
            f"{input_path} has the wrong number of mappings.",
        )


class TestParseExplicit(unittest.TestCase):
    """This test case contains explicit tests for parsing."""

    def _basic_round_trip(self, key: str):
        """Test TSV => JSON => TSV using convert() + parse()."""
        parse_func = PARSING_FUNCTIONS[key]
        write_func, _write_format = WRITER_FUNCTIONS[key]

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
        self.assertEqual(explicit_prefixes.union(SSSOM_BUILT_IN_PREFIXES), set(msdf.prefix_map))

        #: A more explicit definition of what should be in the bijective
        #: prefix map
        expected_bimap = {
            "DOID": "http://purl.obolibrary.org/obo/DOID_",
            "UMLS": "https://uts.nlm.nih.gov/uts/umls/concept/",
            "orcid": "https://orcid.org/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "semapv": "https://w3id.org/semapv/vocab/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "sssom": "https://w3id.org/sssom/",
        }
        self.assertEqual(expected_bimap, msdf.converter.bimap)

        with TemporaryDirectory() as directory:
            path = Path(directory).joinpath("test.sssom.x")
            with path.open("w") as file:
                write_func(msdf, file)

            reconstituted_msdf = parse_func(path)
            reconstituted_msdf.clean_prefix_map(strict=True)

            test_meta = {
                "mapping_set_title": "A title",
                "license": "https://w3id.org/sssom/license/test",
            }
            reconstituted_msdf_with_meta = parse_func(path, meta=test_meta)
            reconstituted_msdf_with_meta.clean_prefix_map(strict=True)

        # Ensure the prefix maps are equal after json parsing and cleaning
        self.assertEqual(
            set(expected_bimap),
            set(reconstituted_msdf.prefix_map),
            msg="Reconstituted prefix map has different CURIE prefixes",
        )
        self.assertEqual(
            expected_bimap,
            reconstituted_msdf.prefix_map,
            msg="Reconstituted prefix map has different URI prefixes",
        )

        # Ensure the shape, labels, and values in the data frame are the same after json parsing and cleaning
        self.assertTrue(msdf.df.equals(reconstituted_msdf.df))

        # Ensure the metadata is the same after json parsing and cleaning
        self.assertEqual(msdf.metadata, reconstituted_msdf.metadata)

        combine_meta = dict(ChainMap(msdf.metadata, test_meta))

        # Ensure the metadata after json parsing with additional metadata corresponds to
        # a chain of the original metadata with the test metadata.
        # In particular, this ensures that fields in the test metadata provided are added
        # to the MappingSet if they are not present, but not updated if they are already present.
        self.assertEqual(combine_meta, reconstituted_msdf_with_meta.metadata)

    def test_round_trip_json(self):
        """Test writing then reading JSON."""
        self._basic_round_trip("json")

    def test_round_trip_rdf(self):
        """Test writing then reading RDF."""
        self._basic_round_trip("rdf")

    def test_round_trip_tsv(self):
        """Test writing then reading TSV."""
        self._basic_round_trip("tsv")

    def test_strict_parsing(self):
        """Test Strict parsing mode."""
        input_path = f"{test_data_dir}/basic_strict_fail.tsv"
        with open(input_path, "r") as file:
            input_string = file.read()
        stream = io.StringIO(input_string)

        with self.assertRaises(ValueError):
            parse_sssom_table(stream, strict=True)

        # Make sure it parses in non-strict mode
        msdf = parse_sssom_table(stream)
        self.assertEqual(len(msdf.df), 2)

    def test_check_irregular_metadata(self):
        """Test if irregular metadata check works according to https://w3id.org/sssom/spec."""
        meta_fail_because_undeclared_extension = {
            "licenses": "http://licen.se",
            "mapping_set_id": "http://mapping.set/id1",
            "ext_test": "value",
        }
        meta_fail_because_extension_without_property = {
            "license": "http://licen.se",
            "mapping_set_id": "http://mapping.set/id1",
            "ext_test": "value",
            "extension_definitions": [{"slot_name": "ext_test"}],
        }

        meta_ok = {
            "license": "http://licen.se",
            "mapping_set_id": "http://mapping.set/id1",
            "ext_test": "value",
            "extension_definitions": [
                {"slot_name": "ext_test", "property": "skos:fantasyRelation"}
            ],
        }

        from sssom.parsers import _is_check_valid_extension_slot, _is_irregular_metadata

        is_irregular_metadata_fail_undeclared_case = _is_irregular_metadata(
            [meta_fail_because_undeclared_extension]
        )
        is_valid_extension = _is_check_valid_extension_slot("ext_test", meta_ok)
        is_irregular_metadata_ok_case = _is_irregular_metadata([meta_ok])
        is_irregular_metadata_fail_missing_property_case = _is_irregular_metadata(
            [meta_fail_because_extension_without_property]
        )
        self.assertTrue(is_irregular_metadata_fail_undeclared_case)
        self.assertTrue(is_irregular_metadata_fail_missing_property_case)
        self.assertTrue(is_valid_extension)
        self.assertFalse(is_irregular_metadata_ok_case)
