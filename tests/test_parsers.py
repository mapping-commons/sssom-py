import json
import os
import unittest
from xml.dom import minidom

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
    read_sssom_table,
)
from sssom.writers import write_table
from tests.test_data import test_data_dir, test_out_dir

cwd = os.path.abspath(os.path.dirname(__file__))


class TestParse(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists(test_out_dir):
            os.mkdir(test_out_dir)

        self.df_url = "https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic.tsv"

        self.rdf_graph_file = f"{test_data_dir}/basic.sssom.rdf"
        self.rdf_graph = Graph()
        self.rdf_graph.parse(self.rdf_graph_file, format="ttl")

        self.df_file = f"{test_data_dir}/basic-meta-external.tsv"
        self.df = pd.read_csv(self.df_file)

        self.obographs_file = f"{test_data_dir}/pato.json"
        with open(self.obographs_file) as json_file:
            self.obographs = json.load(json_file)

        self.json_file = f"{test_data_dir}/basic.json"
        with open(self.json_file) as json_file:
            self.json = json.load(json_file)

        with open(f"{test_data_dir}/basic-meta-external.yml") as file:
            df_meta = yaml.safe_load(file)
            self.df_curie_map = df_meta["curie_map"]
            self.df_meta = df_meta
            self.df_meta.pop("curie_map", None)

        self.alignmentxml_file = f"{test_data_dir}/oaei-ordo-hp.rdf"
        self.alignmentxml = minidom.parse(self.alignmentxml_file)
        self.metadata, self.curie_map = get_default_metadata()

    def test_parse_sssom_dataframe(self):
        file = f"{test_data_dir}/basic.tsv"
        msdf = read_sssom_table(file)
        write_table(msdf, os.path.join(test_out_dir, "test_parse_sssom_dataframe.tsv"))
        self.assertEqual(
            len(msdf.df),
            141,
            f"{file} has the wrong number of mappings.",
        )

    def test_parse_sssom_dataframe_url(self):
        msdf = read_sssom_table(self.df_url)
        write_table(
            msdf, os.path.join(test_out_dir, "test_parse_sssom_dataframe_url.tsv")
        )
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.df_url} has the wrong number of mappings.",
        )

    def test_parse_obographs(self):
        msdf = from_obographs(
            jsondoc=self.obographs, curie_map=self.curie_map, meta=self.metadata
        )
        write_table(msdf, os.path.join(test_out_dir, "test_parse_obographs.tsv"))
        self.assertEqual(
            len(msdf.df),
            9941,
            f"{self.obographs_file} has the wrong number of mappings.",
        )

    def test_parse_tsv(self):
        msdf = from_sssom_dataframe(
            df=self.df, curie_map=self.df_curie_map, meta=self.df_meta
        )
        write_table(msdf, os.path.join(test_out_dir, "test_parse_tsv.tsv"))
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.df_file} has the wrong number of mappings.",
        )

    def test_parse_alignment_minidom(self):
        msdf = from_alignment_minidom(
            dom=self.alignmentxml, curie_map=self.curie_map, meta=self.metadata
        )
        write_table(
            msdf, os.path.join(test_out_dir, "test_parse_alignment_minidom.tsv")
        )
        self.assertEqual(
            len(msdf.df),
            646,
            f"{self.alignmentxml_file} has the wrong number of mappings.",
        )

    def test_parse_sssom_rdf(self):
        msdf = from_sssom_rdf(
            g=self.rdf_graph, curie_map=self.df_curie_map, meta=self.metadata
        )
        write_table(msdf, os.path.join(test_out_dir, "test_parse_sssom_rdf.tsv"))
        self.assertEqual(
            len(msdf.df),
            136,
            f"{self.rdf_graph_file} has the wrong number of mappings.",
        )

    def test_parse_sssom_json(self):
        msdf = from_sssom_json(
            jsondoc=self.json, curie_map=self.df_curie_map, meta=self.metadata
        )
        write_table(msdf, os.path.join(test_out_dir, "test_parse_sssom_json.tsv"))
        self.assertEqual(
            len(msdf.df),
            141,
            f"{self.json_file} has the wrong number of mappings.",
        )
