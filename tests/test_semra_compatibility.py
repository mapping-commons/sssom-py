"""Test for merging MappingSetDataFrames."""

import unittest

from sssom_schema import Mapping

from sssom.context import get_converter
from sssom.parsers import parse_sssom_table
from sssom.util import MappingSetDataFrame
from sssom.writers import write_table


class TestSemraCompatibility(unittest.TestCase):
    """A test case for making sure the model works as intended."""

    def test_basic_inference(self):
        """Test if instantiating Mapping() fails when required elements are missing."""
        mdict_missing = dict(
            subject_id="ID:123"
        )  # This is missing object_id, predicate_id, mapping_justification

        import io

        import pandas as pd
        from semra.api import infer_chains, infer_reversible
        from semra.io import from_sssom_df, get_sssom_df

        data = [
            ["UBERON:1", "skos:exactMatch", "FBbt:9"],
            ["UBERON:1", "skos:exactMatch", "WBbt:6"],
        ]

        df = pd.DataFrame(data=data, columns=["subject_id", "predicate_id", "object_id"])

        mappings = from_sssom_df(df, mapping_set_name="test")
        mappings = infer_reversible(mappings, progress=False)
        mappings = infer_chains(mappings, progress=False)

        df = get_sssom_df(mappings)
        print(df)
        msdf = MappingSetDataFrame(df=df, converter=get_converter())
        print(msdf.df)
        msdf.standardize_references()
        msdf.clean_prefix_map()
        with open("testout.sssom.tsv", "w", encoding="utf-8") as file:
            write_table(msdf=msdf, file=file)
