"""Test s-expressions."""

import re
import unittest
from pathlib import Path

import pandas as pd
from curies import Converter

import sssom.io
from sssom import Mapping
from sssom.sexpr import get_mapping_hash, to_sexpr

HERE = Path(__file__).parent.resolve()
PATH = HERE.joinpath("data", "sexpr_test.sssom.tsv")


class TestSExpressions(unittest.TestCase):
    """Test creation of canonical S-expressions."""

    def test_explicit_example(self) -> None:
        """Test a hard-coded example, explicit in the code."""
        converter = Converter.from_prefix_map(
            {
                "FBbt": "http://purl.obolibrary.org/obo/FBbt_",
                "UBERON": "http://purl.obolibrary.org/obo/UBERON_",
                "orcid": "https://orcid.org/",
                "semapv": "https://w3id.org/semapv/vocab/",
                "skos": "http://www.w3.org/2004/02/skos/core#",
            }
        )
        sexpr = """
        (7:mapping(
           (10:subject_id44:http://purl.obolibrary.org/obo/FBbt_00001234)
           (12:predicate_id46:http://www.w3.org/2004/02/skos/core#exactMatch)
           (9:object_id45:http://purl.obolibrary.org/obo/UBERON_0005678)
           (21:mapping_justification51:https://w3id.org/semapv/vocab/ManualMappingCuration)
           (10:creator_id(
                          37:https://orcid.org/0000-0000-1234-5678
                          37:https://orcid.org/0000-0000-5678-1234
            ))
        ))
        """
        mapping = Mapping(
            subject_id="http://purl.obolibrary.org/obo/FBbt_00001234",
            predicate_id="http://www.w3.org/2004/02/skos/core#exactMatch",
            object_id="http://purl.obolibrary.org/obo/UBERON_0005678",
            mapping_justification="https://w3id.org/semapv/vocab/ManualMappingCuration",
            creator_id=[
                "https://orcid.org/0000-0000-1234-5678",
                "https://orcid.org/0000-0000-5678-1234",
            ],
        )
        self.assertEqual(re.sub(r"\s", "", sexpr), to_sexpr(mapping, converter))
        self.assertEqual(
            "hq6bs14aptzepwgk6pw7j8ysnft6riqrw7har84rtk8r9xmbcwty",
            get_mapping_hash(mapping, converter),
        )

    def test_all(self) -> None:
        """Test all."""
        msdf = sssom.parse_tsv(PATH)

        # After new SSSOM schema release, this will be part of the mapping data model
        record_ids = pd.read_csv(PATH, sep="\t", skiprows=5)["record_id"]
        for record_id, mapping in zip(record_ids, msdf.to_mappings()):
            self.assertEqual(
                record_id.removeprefix("sssom.record:"),
                get_mapping_hash(mapping, msdf.converter),
                msg=to_sexpr(mapping, msdf.converter),
            )
