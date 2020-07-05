from sssom.io import from_tsv, to_rdf
from sssom import Mapping, MappingSet
from json import loads, dumps
from jsonasobj import as_json_obj, as_json
from biolinkml.utils.yamlutils import DupCheckYamlLoader, as_json_object as yaml_to_json
import json
from rdflib import Graph
from biolinkml.generators.jsonldcontextgen import ContextGenerator

import unittest
import os

import logging

cwd = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(cwd, 'data')
schema_dir = os.path.join(cwd, '../schema')

class TestParse(unittest.TestCase):

    def setUp(self) -> None:
        print('Parsing...')
        self.mdoc = from_tsv(f'{data_dir}/basic.tsv')

    def test_ms(self):
        ms = self.mdoc.mapping_set
        #for m in ms.mappings:
        #    print(f'M={m}')
        self.assertTrue(True)
        #g = to_rdf(self.mdoc, context_path=f'{schema_dir}/sssom.context.jsonld')
        g = to_rdf(self.mdoc)
        print(g.serialize(format="turtle").decode())

if __name__ == '__main__':
    unittest.main()
