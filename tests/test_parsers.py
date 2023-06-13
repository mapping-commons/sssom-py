"""Tests for parsers."""

import json
import math
import os
import unittest
from xml.dom import minidom

import numpy as np
import pandas as pd
import yaml
from rdflib import Graph
from sssom.context import get_default_metadata
from sssom.parsers import (
    from_alignment_minidom,
    from_obographs,
    from_sssom_dataframe,
    from_sssom_json,
    from_sssom_rdf,
    parse_sssom_table,
)
from sssom.util import PREFIX_MAP_KEY, sort_df_rows_columns
from sssom.writers import write_table

from tests.test_data import data_dir as test_data_dir
from tests.test_data import test_out_dir


class TestParse(unittest.TestCase):
    """A test case for parser functionality."""

    def setUp(self) -> None:
        """Set up the test case."""
        # TODO: change back to the commented url.
        self.df_url = "https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic.tsv"
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
            self.df_prefix_map = df_meta.pop(PREFIX_MAP_KEY)
            self.df_meta = df_meta

        self.alignmentxml_file = f"{test_data_dir}/oaei-ordo-hp.rdf"
        self.alignmentxml = minidom.parse(self.alignmentxml_file)
        self.metadata = get_default_metadata()

    def test_parse_sssom_dataframe(self):
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

    def test_parse_sssom_dataframe_url(self):
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
            prefix_map=self.metadata.prefix_map,
            meta=self.metadata.metadata,
        )
        path = os.path.join(test_out_dir, "test_parse_obographs.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            9880,
            f"{self.obographs_file} has the wrong number of mappings.",
        )

    def test_broken_obographs(self):
        """Test parsing OBO Graph JSON."""
        prefix_map = self.metadata.prefix_map
        prefix_map["OMIM"] = "http://omim.org/entry/"
        with self.assertRaises(ValueError):
            from_obographs(
                jsondoc=self.broken_obographs,
                prefix_map=prefix_map,
                meta=self.metadata.metadata,
            )

    def test_parse_tsv(self):
        """Test parsing TSV."""
        msdf = from_sssom_dataframe(
            df=self.df, prefix_map=self.df_prefix_map, meta=self.df_meta
        )
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
            prefix_map=self.metadata.prefix_map,
            meta=self.metadata.metadata,
        )
        path = os.path.join(test_out_dir, "test_parse_alignment_minidom.tsv")
        with open(path, "w") as file:
            write_table(msdf, file)
        self.assertEqual(
            len(msdf.df),
            646,
            f"{self.alignmentxml_file} has the wrong number of mappings.",
        )

    def test_parse_sssom_rdf(self):
        """Test parsing RDF."""
        msdf = from_sssom_rdf(
            g=self.rdf_graph, prefix_map=self.df_prefix_map, meta=self.metadata.metadata
        )
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
            prefix_map=self.df_prefix_map,
            meta=self.metadata.metadata,
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
