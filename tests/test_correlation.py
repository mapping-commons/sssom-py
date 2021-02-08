import json
import os
import unittest

import click
from click.testing import CliRunner

from rdflib import Graph

from sssom.parsers import from_tsv, from_rdf, from_alignment_xml, from_owl, read_pandas
from sssom.writers import to_owl_graph, to_rdf_graph, to_dataframe, to_jsonld_dict
from sssom.writers import write_json, write_rdf, write_owl, write_tsv
from sssom import slots
from sssom.cli import correlations

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
        self.mdoc = from_tsv(_data("bosch-wd-matches"))
        self.context_path = f'{schema_dir}/sssom.context.jsonld'
        curie_map = {}
        curie_map["HP"] = "http://purl.obolibrary.org/obo/HP_"
        curie_map["ORDO"] = "http://www.orpha.net/ORDO/Orphanet_"
        self.curie_map = curie_map
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)

    def test_correlation(self):
        runner = CliRunner()
        result = runner.invoke(correlations, _data("bosch-wd-matches"), f'{test_out_dir}/"bosch-wd-category-pairs-chisquared.tsv')
        assert(True)


if __name__ == '__main__':
    unittest.main()
