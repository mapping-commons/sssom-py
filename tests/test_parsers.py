import json
import os
import unittest
from xml.dom import minidom
from hbreader import FileInfo, hbread

import pandas as pd
import yaml
from rdflib import Graph

from sssom.context import get_default_metadata
from sssom.parsers import read_sssom_table, from_obographs, from_sssom_dataframe, from_alignment_minidom, from_sssom_rdf, \
    from_sssom_json
from sssom.writers import write_tsv
# from pandasql import sqldf
from tests.test_data import test_out_dir, test_data_dir

cwd = os.path.abspath(os.path.dirname(__file__))


class TestParse(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists(test_out_dir):
            os.mkdir(test_out_dir)
        self.rdf_graph = Graph()
        self.rdf_graph.parse(f"{test_data_dir}/basic.sssom.rdf", format="ttl")
        self.df = pd.read_csv(f"{test_data_dir}/basic-meta-external.tsv")
        with open(f"{test_data_dir}/pato.json") as json_file:
            self.obographs = json.load(json_file)
        self.json = hbread(f"{test_data_dir}/basic.json")
        with open(f"{test_data_dir}/basic-meta-external.yml") as file:
            self.df_meta = yaml.load(file, Loader=yaml.FullLoader)
        with open(f"{test_data_dir}/basic-meta-external.yml") as file:
            df_meta = yaml.load(file, Loader=yaml.FullLoader)
            self.df_curie_map = df_meta['curie_map']
            self.df_meta = df_meta
            self.df_meta.pop('curie_map', None)
        self.alignmentxml = minidom.parse(f"{test_data_dir}/oaei-ordo-hp.rdf")
        self.metadata, self.curie_map = get_default_metadata()

    def test_parse_sssom_dataframe(self):
        msdf = read_sssom_table(f"{test_data_dir}/basic.tsv")
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_tsv.tsv"))
        self.assertEqual(
            len(msdf.df),
            141,
            f"basic.tsv has the wrong number of mappings.",
        )

    def test_parse_obographs(self):
        msdf = from_obographs(jsondoc=self.obographs, curie_map=self.curie_map, meta=self.metadata)
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_obographs.tsv"))
        self.assertEqual(
            len(msdf.df),
            9941,
            f"pato.jsom has the wrong number of mappings.",
        )

    def test_parse_tsv(self):
        msdf = from_sssom_dataframe(df=self.df, curie_map=self.df_curie_map, meta=self.df_meta)
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_tsv.tsv"))
        self.assertEqual(
            len(msdf.df),
            141,
            f"basic-no-merge.tsv has the wrong number of mappings.",
        )

    def test_parse_alignment_minidom(self):
        msdf = from_alignment_minidom(dom=self.alignmentxml, curie_map=self.curie_map, meta=self.metadata)
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_alignment_minidom.tsv"))
        self.assertEqual(
            len(msdf.df),
            646,
            f"basic-no-merge.tsv has the wrong number of mappings.",
        )

    def test_parse_sssom_rdf(self):
        msdf = from_sssom_rdf(g=self.rdf_graph, curie_map=self.curie_map, meta=self.metadata)
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_sssom_rdf.tsv"))
        self.assertEqual(
            len(msdf.df),
            646,
            f"basic-no-merge.tsv has the wrong number of mappings.",
        )

    def test_parse_sssom_json(self):
        msdf = from_sssom_json(jsondoc=self.json, curie_map=self.curie_map, meta=self.metadata)
        write_tsv(msdf, os.path.join(test_out_dir, "test_parse_sssom_json.tsv"))
        self.assertEqual(
            len(msdf.df),
            646,
            f"basic-no-merge.tsv has the wrong number of mappings.",
        )