import json
import os
import unittest

from rdflib import Graph

from sssom.parsers import from_tsv, from_rdf, from_alignment_xml, from_owl, read_pandas
from sssom.writers import to_owl_graph, to_rdf_graph, to_dataframe, to_jsonld_dict
from sssom.writers import write_json, write_rdf, write_owl, write_tsv

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, 'data')
test_out_dir = os.path.join(cwd, 'tmp')
schema_dir = os.path.join(cwd, '../schema')

query1 = """
SELECT DISTINCT ?e1 ?e2
WHERE {
  ?e1 owl:equivalentClass ?e2 .
}"""


def _data(filename, ext="tsv"):
    return f'{data_dir}/{filename}.{ext}'


class TestParse(unittest.TestCase):

    def setUp(self) -> None:
        print('Parsing...')
        self.mdoc = from_tsv(_data("basic"))
        self.context_path = f'{schema_dir}/sssom.context.jsonld'
        curie_map = {}
        curie_map["HP"] = "http://purl.obolibrary.org/obo/HP_"
        curie_map["ORDO"] = "http://www.orpha.net/ORDO/Orphanet_"
        self.curie_map = curie_map
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)

    def test_to_owl_graph(self):
        g = to_owl_graph(self.mdoc)
        self.assertEqual(len(g.query(query1)), 86)

    def test_to_rdf_graph(self):
        self.assertTrue(True)
        g = to_rdf_graph(self.mdoc)
        self.assertEqual(len(g.query(query1)), 86)

    def test_to_dataframe(self):
        df = to_dataframe(self.mdoc)
        self.assertEqual(len(df), 136)

    def test_to_json_dict(self):
        self.assertTrue(True)
        json_dict = to_jsonld_dict(self.mdoc)
        with open(f'{test_out_dir}/basic.jsonld', 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
        self.assertEqual(len(json_dict), 176)

    def test_write_owl(self):
        fn = f'{test_out_dir}/basic.owl'
        write_owl(self.mdoc, fn)
        g = Graph()
        g.parse(fn, format='xml')
        self.assertEqual(len(g.query(query1)), 86)

    def test_write_rdf(self):
        fn = f'{test_out_dir}/basic.owl'
        write_rdf(self.mdoc, fn)
        g = Graph()
        g.parse(fn, format='xml')
        self.assertEqual(len(g.query(query1)), 86)

    def test_write_tsv(self):
        fn = f'{test_out_dir}/basic.tsv'
        write_tsv(self.mdoc, fn)
        df = read_pandas(fn)
        self.assertEqual(len(df), 136)

    def test_write_jsonld(self):
        fn = f'{test_out_dir}/basic.tsv'
        write_json(self.mdoc, fn)
        with open(fn) as json_file:
            data = json.load(json_file)
        self.assertEqual(len(data), 176)

    def test_from_tsv(self):
        ms = from_tsv(_data("cob-to-external"))
        self.assertEqual(len(ms.mapping_set.mappings), 104)
        fn = f'{test_out_dir}/cob-to-external.owl'
        write_owl(ms, fn)

    #
    # def test_from_rdf(self):
    #     ms = from_rdf(_data("basic", "ttl"), curie_map=self.curie_map)
    #     self.assertEqual(len(ms.mapping_set.mappings), 136)
    #
    # def test_from_owl(self):
    #     ms = from_owl(_data("basic", "owl"), curie_map=self.curie_map)
    #     self.assertEqual(len(ms.mapping_set.mappings), 136)

    def test_from_alignment_format(self):
        ms = from_alignment_xml(_data("oaei-ordo-hp", "rdf"), self.curie_map)
        self.assertEqual(len(ms.mapping_set.mappings), 646)
        fn = f'{test_out_dir}/oaei-ordo-hp.tsv'
        write_tsv(ms, fn)
        df = read_pandas(fn)
        self.assertEqual(len(df), 646)


if __name__ == '__main__':
    unittest.main()
