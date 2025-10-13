"""Test for merging MappingSetDataFrames."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from curies import Converter, Record, chain
from sssom_schema import Mapping as SSSOM_Mapping
from sssom_schema import slots as SSSOM_Slots

from sssom.constants import (
    CARDINALITY_SCOPE,
    CREATOR_ID,
    MAPPING_CARDINALITY,
    MAPPING_SET_CONFIDENCE,
    MAPPING_TOOL_ID,
    OBJECT_ID,
    OBJECT_LABEL,
    OBJECT_TYPE,
    PREDICATE_ID,
    PREDICATE_TYPE,
    SEMAPV,
    SUBJECT_ID,
    SUBJECT_LABEL,
    SUBJECT_TYPE,
    MetadataType,
)
from sssom.context import SSSOM_BUILT_IN_PREFIXES, ensure_converter
from sssom.io import extract_iris
from sssom.parsers import parse_sssom_table
from sssom.util import (
    MappingSetDataFrame,
    filter_out_prefixes,
    filter_prefixes,
    get_dict_from_mapping,
    get_file_extension,
    get_prefixes_used_in_table,
    invert_mappings,
    is_multivalued_slot,
)
from tests.constants import data_dir

HERE = Path(__file__).parent.resolve()


class TestIO(unittest.TestCase):
    """A test case for merging msdfs."""

    def setUp(self) -> None:
        """Set up."""
        self.msdf = parse_sssom_table(f"{data_dir}/basic.tsv")
        self.msdf2 = parse_sssom_table(f"{data_dir}/basic7.tsv")
        self.features = [SUBJECT_ID, OBJECT_ID]
        self.mapping_justification = SEMAPV.ManualMappingCuration.value

    def test_broken_predicate_list(self) -> None:
        """Test merging of multiple msdfs."""
        predicate_filter = ["skos:relatedMatch", [f"{data_dir}/predicate_list3.txt"]]
        prefix_map = {"skos": "http://www.w3.org/2004/02/skos/core#"}
        converter = Converter.from_prefix_map(prefix_map)
        iri_list = extract_iris(predicate_filter, converter=converter)
        self.assertEqual(
            [
                "http://www.w3.org/2004/02/skos/core#narrowMatch",
                "http://www.w3.org/2004/02/skos/core#relatedMatch",
            ],
            iri_list,
        )

    def test_filter_prefixes_any(self) -> None:
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_prefixes(
            original_msdf.df,
            prefix_filter_list,
            self.features,
            require_all_prefixes=False,
        )
        self.assertEqual(len(filtered_df), 136)

    def test_filter_prefixes_all(self) -> None:
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_prefixes(
            original_msdf.df,
            prefix_filter_list,
            self.features,
            require_all_prefixes=True,
        )
        self.assertEqual(len(filtered_df), 40)

    def test_filter_out_prefixes_any(self) -> None:
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(
            original_msdf.df,
            prefix_filter_list,
            self.features,
            require_all_prefixes=False,
        )
        self.assertEqual(len(filtered_df), 5)

    def test_filter_out_prefixes_all(self) -> None:
        """Test filtering MSDF.df by prefixes provided."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(
            original_msdf.df,
            prefix_filter_list,
            self.features,
            require_all_prefixes=True,
        )
        self.assertEqual(len(filtered_df), 101)

    def test_remove_mappings(self) -> None:
        """Test remove mappings."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(original_msdf.df, prefix_filter_list, self.features)
        new_msdf = MappingSetDataFrame.with_converter(
            df=filtered_df,
            converter=original_msdf.converter,
            metadata=original_msdf.metadata,
        )
        original_length = len(original_msdf.df)
        original_msdf.remove_mappings(new_msdf)
        # len(self.msdf.df) = 141 and len(new_msdf.df) = 5
        self.assertEqual(len(original_msdf.df), original_length - len(new_msdf.df))

    def test_clean_prefix_map(self) -> None:
        """Test clean prefix map."""
        prefix_filter_list = ["x", "y"]
        original_msdf = self.msdf
        filtered_df = filter_out_prefixes(original_msdf.df, prefix_filter_list, self.features)
        new_msdf = MappingSetDataFrame.with_converter(
            df=filtered_df,
            converter=original_msdf.converter,
            metadata=original_msdf.metadata,
        )
        new_msdf.clean_prefix_map()
        self.assertEqual(
            new_msdf.prefix_map.keys(),
            set(original_msdf.prefix_map.keys()).intersection(set(new_msdf.prefix_map.keys())),
        )

    def test_clean_prefix_map_strict(self) -> None:
        """Test clean prefix map with 'strict'=True."""
        msdf = parse_sssom_table(f"{data_dir}/test_clean_prefix.tsv")
        with self.assertRaises(ValueError):
            msdf.clean_prefix_map(strict=True)

    def test_clean_prefix_map_not_strict(self) -> None:
        """Test clean prefix map with 'strict'=False."""
        msdf = parse_sssom_table(f"{data_dir}/test_clean_prefix.tsv")
        self.assertEqual(
            {
                "a": "http://example.org/a/",
                "b": "http://example.org/b/",
                "c": "http://example.org/c/",
                "d": "http://example.org/d/",
                "orcid": "https://orcid.org/",
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "semapv": "https://w3id.org/semapv/vocab/",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "sssom": "https://w3id.org/sssom/",
            },
            msdf.prefix_map,
        )
        original_curie_map = msdf.prefix_map
        self.assertEqual(
            {"a", "b", "c", "d", "orcid"}.union(SSSOM_BUILT_IN_PREFIXES),
            set(original_curie_map),
        )
        msdf.clean_prefix_map(strict=False)
        new_curie_map = msdf.prefix_map
        self.assertEqual(
            {"a", "b", "c", "d", "orcid", "x1", "y1", "z1"}.union(SSSOM_BUILT_IN_PREFIXES),
            set(new_curie_map),
        )

    def test_invert_nodes(self) -> None:
        """Test invert nodes."""
        subject_prefix = "a"
        inverted_df = invert_mappings(self.msdf2.df, subject_prefix, False)
        self.assertEqual(len(inverted_df), 13)

    def test_invert_nodes_merged(self) -> None:
        """Test invert nodes with merge_inverted."""
        subject_prefix = "a"
        inverted_df = invert_mappings(self.msdf2.df, subject_prefix, True)
        self.assertEqual(len(inverted_df), 38)

    def test_invert_nodes_without_prefix(self) -> None:
        """Test invert nodes."""
        inverted_df = invert_mappings(df=self.msdf2.df, merge_inverted=False)
        self.assertEqual(len(inverted_df), len(self.msdf2.df.drop_duplicates()))

    def test_invert_asymmetric_nodes(self) -> None:
        """Test inverting sets containing imbalanced subject/object columns."""
        msdf = parse_sssom_table(f"{data_dir}/asymmetric.tsv")
        inverted_df = invert_mappings(msdf.df, merge_inverted=False)
        self.assertEqual(len(inverted_df), len(msdf.df))
        original_subject_labels = msdf.df["subject_label"].values
        inverted_object_labels = inverted_df["object_label"].values
        self.assertNotIn(False, original_subject_labels == inverted_object_labels)


class TestUtils(unittest.TestCase):
    """Unit test for utility functions."""

    def test_get_prefixes(self) -> None:
        """Test getting prefixes from a MSDF."""
        path = data_dir.joinpath("enm_example.tsv")
        metadata_path = data_dir.joinpath("enm_example.yml")
        metadata = yaml.safe_load(metadata_path.read_text())
        msdf = parse_sssom_table(path, meta=metadata)
        prefixes = get_prefixes_used_in_table(msdf.df)
        self.assertNotIn("http", prefixes)
        self.assertNotIn("https", prefixes)
        self.assertEqual(
            {
                "ENVO",
                "NPO",
                "ENM",
                "CHEBI",
                "OBI",
                "IAO",
                "BAO",
                "EFO",
                "BTO",
                "orcid",
            }.union(SSSOM_BUILT_IN_PREFIXES),
            prefixes,
        )

    def test_standardize_df(self) -> None:
        """Test standardizing a MSDF's dataframe."""
        rows = [("a:1", "b:2", "c:3")]
        columns = ["subject_id", "predicate_id", "object_id"]
        df = pd.DataFrame(rows, columns=columns)
        converter = Converter(
            [
                Record(prefix="new.a", prefix_synonyms=["a"], uri_prefix="https://example.org/a/"),
                Record(prefix="new.b", prefix_synonyms=["b"], uri_prefix="https://example.org/b/"),
                Record(prefix="new.c", prefix_synonyms=["c"], uri_prefix="https://example.org/c/"),
            ]
        )
        msdf = MappingSetDataFrame(df=df, converter=converter)
        msdf._standardize_df_references()
        self.assertEqual(
            ("new.a:1", "new.b:2", "new.c:3"),
            tuple(df.iloc[0]),
        )

    def test_standardize_idempotent(self) -> None:
        """Test standardizing leaves correct fields."""
        metadata = {"license": "https://example.org/test-license"}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual({"license": "https://example.org/test-license"}, msdf.metadata)

    def test_standardize_metadata_upgrade_multivalued_single(self) -> None:
        """Test standardizing upgrades a string to a list."""
        metadata = {"creator_id": "orcid:0000-0003-4423-4370"}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual({"creator_id": ["orcid:0000-0003-4423-4370"]}, msdf.metadata)

    def test_standardize_metadata_upgrade_multivalued_multiple(self) -> None:
        """Test standardizing upgrades a string to a list."""
        metadata = {"creator_id": "orcid:0000-0003-4423-4370 | orcid:0000-0002-6601-2165"}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual(
            {"creator_id": ["orcid:0000-0003-4423-4370", "orcid:0000-0002-6601-2165"]},
            msdf.metadata,
        )

    def test_standardize_metadata_multivalued(self) -> None:
        """Test standardizing upgrades a string to a list."""
        metadata = {"creator_id": ["orcid:0000-0003-4423-4370", "orcid:0000-0002-6601-2165"]}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual(
            {"creator_id": ["orcid:0000-0003-4423-4370", "orcid:0000-0002-6601-2165"]},
            msdf.metadata,
        )

    def test_standardize_metadata_multivalued_type_error(self) -> None:
        """Test raising on a non-string, non-list object given to a multivalued slot."""
        metadata = {"creator_id": object()}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        with self.assertRaises(TypeError):
            msdf._standardize_metadata_references(raise_on_invalid=True)

    def test_standardize_error_non_list(self) -> None:
        """Test that a type error is raised when metadata is presented that should not be a list."""
        metadata = {"mapping_source": ["https://example.org/r1", "https://example.org/r2"]}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        with self.assertRaises(TypeError):
            msdf._standardize_metadata_references(raise_on_invalid=True)

    def test_standardize_coerce_list(self) -> None:
        """Test that a metadata field that should not be a list gets coerced to a single element."""
        self.assertFalse(is_multivalued_slot("mapping_source"))
        metadata = {"mapping_source": ["https://example.org/r1"]}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertIn("mapping_source", msdf.metadata)
        mapping_source = msdf.metadata["mapping_source"]
        self.assertIsInstance(
            mapping_source, str, msg="Mapping source should have been coerced into a string"
        )
        self.assertEqual({"mapping_source": "https://example.org/r1"}, msdf.metadata)

    def test_standardize_metdata_delete_empty(self) -> None:
        """Test that an element that should not be a list but is empty just gets deleted."""
        metadata: MetadataType = {"mapping_source": []}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual({}, msdf.metadata)

    def test_standardize_metadata_single(self) -> None:
        """Test standardizing a single valued slot."""
        metadata: MetadataType = {"mapping_source": "https://example.org/r1"}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        msdf._standardize_metadata_references(raise_on_invalid=True)
        self.assertEqual({"mapping_source": "https://example.org/r1"}, msdf.metadata)

    def test_standardize_metadata_raise_on_missing(self) -> None:
        """Test that an exception is raised for a metadata key that's not in the schema."""
        metadata: MetadataType = {"xxxx": "https://example.org/r1"}
        msdf = MappingSetDataFrame(df=pd.DataFrame(), converter=Converter([]), metadata=metadata)
        with self.assertRaises(ValueError):
            msdf._standardize_metadata_references(raise_on_invalid=True)

        # Test the logs actually get through
        with self.assertLogs("sssom.util") as cm:
            msdf._standardize_metadata_references()
            self.assertIn("invalid metadata key xxxx", "".join(cm.output))

    def test_msdf_from_mappings(self) -> None:
        """Test round tripping to SSSOM classes."""
        rows = [
            (
                "DOID:0050601",
                "ADULT syndrome",
                "skos:exactMatch",
                "UMLS:C1863204",
                "ADULT SYNDROME",
                SEMAPV.ManualMappingCuration.value,
                "orcid:0000-0003-4423-4370",
            )
        ]
        columns = [
            SUBJECT_ID,
            SUBJECT_LABEL,
            PREDICATE_ID,
            OBJECT_ID,
            OBJECT_LABEL,
            SEMAPV.ManualMappingCuration.value,
            CREATOR_ID,
        ]
        df = pd.DataFrame(rows, columns=columns)
        msdf = MappingSetDataFrame(df=df, converter=ensure_converter())
        msdf.clean_prefix_map(strict=True)

        msd = msdf.to_mapping_set_document()

        new_msdf = MappingSetDataFrame.from_mappings(
            mappings=msd.mapping_set.mappings,
            converter=msd.converter,
            metadata={
                "license": msdf.metadata["license"],
                "mapping_set_id": msdf.metadata["mapping_set_id"],
            },
        )

        self.assertEqual(1, len(new_msdf.df.index))
        self.assertEqual(rows[0], tuple(msdf.df.iloc[0]))
        self.assertEqual(new_msdf.metadata, msdf.metadata)

    def test_get_dict_from_mapping(self) -> None:
        """Test getting dict from a SSSOM mapping object or a dictionary."""
        mapping_obj = SSSOM_Mapping(
            subject_id="DOID:0050601",
            predicate_id="skos:exactMatch",
            object_id="UMLS:C1863204",
            mapping_justification=SEMAPV.ManualMappingCuration.value,
            author_id=["orcid:0000-0002-2411-565X", "orcid:0000-0002-7356-1779"],
            confidence=0.5,
        )
        mapping_dict = mapping_obj.__dict__

        expected_result = {
            "subject_id": "DOID:0050601",
            "predicate_id": "skos:exactMatch",
            "object_id": "UMLS:C1863204",
            "mapping_justification": "semapv:ManualMappingCuration",
            "subject_label": "",
            "subject_category": "",
            "predicate_label": "",
            "predicate_modifier": "",
            "object_label": "",
            "object_category": "",
            "author_id": "orcid:0000-0002-2411-565X|orcid:0000-0002-7356-1779",
            "author_label": "",
            "reviewer_id": "",
            "reviewer_label": "",
            "creator_id": "",
            "creator_label": "",
            "license": "",
            "subject_type": "",
            "subject_source": "",
            "subject_source_version": "",
            "object_type": "",
            "object_source": "",
            "object_source_version": "",
            "mapping_provider": "",
            "mapping_source": "",
            "mapping_cardinality": "",
            "mapping_tool": "",
            "mapping_tool_version": "",
            "mapping_date": "",
            "publication_date": "",
            "confidence": 0.5,
            "curation_rule": "",
            "curation_rule_text": "",
            "subject_match_field": "",
            "object_match_field": "",
            "match_string": "",
            "subject_preprocessing": "",
            "object_preprocessing": "",
            "see_also": "",
            "issue_tracker_item": "",
            "other": "",
            "comment": "",
        }

        if hasattr(SSSOM_Slots, "similarity_score"):
            expected_result["similarity_score"] = np.nan
            expected_result["similarity_measure"] = ""
        else:
            expected_result["semantic_similarity_score"] = np.nan
            expected_result["semantic_similarity_measure"] = ""

        result_with_mapping_object = get_dict_from_mapping(mapping_obj)
        result_with_dict = get_dict_from_mapping(mapping_dict)
        self.assertEqual(result_with_mapping_object, result_with_dict)

        # Assert that every attribute value in expected_result
        # equals the corresponding key in result_with_mapping_object (except lists)
        for key, value in expected_result.items():
            if value is None or value == [] or value is np.nan:
                self.assertIn(result_with_mapping_object[key], [np.nan, ""])
                self.assertIn(result_with_dict[key], [np.nan, ""])
            else:
                self.assertEqual(value, result_with_mapping_object[key])
                self.assertEqual(value, result_with_dict[key])

    def test_curiechain_with_conflicts(self) -> None:
        """Test curie map with CURIE/URI clashes."""
        PREFIXMAP_BOTH = {
            "SCTID": "http://identifiers.org/snomedct/",
            "SCTID__2": "http://snomed.info/id/",
        }
        PREFIXMAP_FIRST = {
            "SCTID": "http://identifiers.org/snomedct/",
        }
        PREFIXMAP_SECOND = {
            "SCTID__2": "http://snomed.info/id/",
        }

        EPM = [
            {
                "prefix": "SCTID",
                "prefix_synonyms": ["snomed"],
                "uri_prefix": "http://snomed.info/id/",
            },
        ]

        converter = chain(
            [Converter.from_prefix_map(PREFIXMAP_FIRST), Converter.from_extended_prefix_map(EPM)]
        )
        self.assertIn("SCTID", converter.prefix_map)
        converter = chain(
            [Converter.from_prefix_map(PREFIXMAP_SECOND), Converter.from_extended_prefix_map(EPM)]
        )
        self.assertIn("SCTID", converter.prefix_map)
        # Fails here:
        with self.assertRaises(ValueError):
            chain(
                [Converter.from_prefix_map(PREFIXMAP_BOTH), Converter.from_extended_prefix_map(EPM)]
            )

        # self.assertIn("SCTID", converter.prefix_map)

    def test_get_file_extension(self) -> None:
        """Test getting a file extension."""
        for value, part in [
            ("tsv", "test.tsv"),
            ("tsv", "test.tsv.gz"),
            ("csv", "test.csv"),
            ("csv", "test.csv.gz"),
            # Don't infer an extension for something else
            (None, "test.xxx"),
        ]:
            path = HERE.joinpath(part)
            with self.subTest(path=path, mode="path"):
                self.assertEqual(value, get_file_extension(path))
            with self.subTest(path=path, mode="str"):
                self.assertEqual(value, get_file_extension(path.as_posix()))
            with self.subTest(path=path, mode="file"), tempfile.TemporaryDirectory() as d:
                with Path(d).joinpath(part).open("w") as file:
                    self.assertEqual(value, get_file_extension(file))

    def test_propagation_and_condensation(self) -> None:
        """Test propagating/condensing values of propagatable slots."""
        # Disable parsing-time propagation so that we can test it explicitly
        msdf = parse_sssom_table(f"{data_dir}/propagatable.tsv", propagate=False)

        propagated_slots = msdf.propagate()
        # creator_id is not a propagatable slot
        self.assertNotIn("creator_id", propagated_slots)
        self.assertNotIn("creator_id", msdf.df.columns)
        # mapping_tool has values for some records and should not be propagated
        self.assertNotIn("mapping_tool", propagated_slots)
        # mapping_provider should be propagated
        self.assertIn("mapping_provider", propagated_slots)
        self.assertIn("mapping_provider", msdf.df.columns)
        self.assertNotIn("mapping_provider", msdf.metadata)
        # Ditto for subject_preprocessing
        self.assertIn("subject_preprocessing", propagated_slots)
        self.assertIn("subject_preprocessing", msdf.df.columns)
        self.assertNotIn("subject_preprocessing", msdf.metadata)

        propagated_slots = msdf.propagate()
        # Set has been propagated already, no further propagation possible
        self.assertEqual(0, len(propagated_slots))

        condensed_slots = msdf.condense()
        # mapping_tool has not a unique value and should not be condensed
        self.assertNotIn("mapping_tool", condensed_slots)
        self.assertIn("mapping_tool", msdf.df.columns)
        # mapping_provider should be condensed back
        self.assertIn("mapping_provider", condensed_slots)
        self.assertNotIn("mapping_provider", msdf.df.columns)
        self.assertIn("mapping_provider", msdf.metadata)
        # Ditto for subject_preprocessing
        self.assertIn("subject_preprocessing", condensed_slots)
        self.assertNotIn("subject_preprocessing", msdf.df.columns)
        self.assertIn("subject_preprocessing", msdf.metadata)

        condensed_slots = msdf.condense()
        # Set has been condensed already, no further condensation possible
        self.assertEqual(0, len(condensed_slots))

    def test_condensation_with_existing_set_values(self) -> None:
        """Test that condensation works as expected with the mapping set already contains values for the to-be-condensed slots."""
        msdf = parse_sssom_table(f"{data_dir}/propagatable.tsv")
        # Following propagation, all records in msdf have the same
        # mapping_provider ("https://example.org/mappings/)"

        # Inject a different mapping_provider value in the set metadata;
        # this should prevent that slot from being condensed back
        msdf.metadata["mapping_provider"] = "https://example.org/mappings/2"
        condensed_slots = msdf.condense()
        self.assertNotIn("mapping_provider", condensed_slots)
        self.assertIn("mapping_provider", msdf.df.columns)
        self.assertEqual("https://example.org/mappings/2", msdf.metadata["mapping_provider"])

        # Inject the same mapping_provider value as the one contained in
        # the records; this should allow the slot to be condensed
        msdf.metadata["mapping_provider"] = "https://example.org/mappings"
        condensed_slots = msdf.condense()
        self.assertIn("mapping_provider", condensed_slots)
        self.assertNotIn("mapping_provider", msdf.df.columns)
        self.assertEqual("https://example.org/mappings", msdf.metadata["mapping_provider"])

    def test_propagation_fill_empty_mode(self) -> None:
        """Test propagate with fill_empty=True."""
        msdf = parse_sssom_table(f"{data_dir}/propagatable.tsv")

        propagated_slots = msdf.propagate(fill_empty=True)
        # mapping_tool should have been propagated
        self.assertIn("mapping_tool", propagated_slots)
        self.assertNotIn("mapping_tool", msdf.metadata)
        self.assertEqual(2, len(msdf.df["mapping_tool"].unique()))

    def test_infer_cardinality(self) -> None:
        """Test cardinality computation."""

        def _check_against_precomputed_values(filename: str) -> None:
            msdf = parse_sssom_table(data_dir.joinpath(filename))
            # Expected values are already contained in the test file
            expected = list(msdf.df[MAPPING_CARDINALITY].values)
            msdf.df.drop(columns=MAPPING_CARDINALITY, inplace=True)
            msdf.infer_cardinality()
            self.assertEqual(expected, list(msdf.df[MAPPING_CARDINALITY].values))

        _check_against_precomputed_values("cardinality.sssom.tsv")
        _check_against_precomputed_values("cardinality-with-NoTermFound.sssom.tsv")
        _check_against_precomputed_values("cardinality-with-literal-mappings.sssom.tsv")

    def test_infer_scoped_cardinality(self) -> None:
        """Test cardinality computation with scopes."""
        msdf = parse_sssom_table(f"{data_dir}/cardinality-scope.sssom.tsv")

        msdf.infer_cardinality(["predicate_id"])
        expected = ["1:n", "1:n", "1:n", "1:n", "1:1", "1:1"]
        self.assertEqual(expected, list(msdf.df[MAPPING_CARDINALITY].values))

        msdf.infer_cardinality(["object_source"])
        expected = ["1:1", "1:1", "1:1", "1:1", "1:1", "1:1"]
        self.assertEqual(expected, list(msdf.df[MAPPING_CARDINALITY].values))

        msdf.infer_cardinality(["object_source", "not_a_valid_slot_name"])
        # should yield the same result as above
        self.assertEqual(expected, list(msdf.df[MAPPING_CARDINALITY].values))

        msdf.infer_cardinality(["not_a_valid_slot_name"])
        # should be equivalent to an empty scope
        expected = ["1:n", "1:n", "1:n", "1:n", "1:n", "1:n"]
        self.assertEqual(expected, list(msdf.df[MAPPING_CARDINALITY].values))
        self.assertNotIn(CARDINALITY_SCOPE, msdf.df.columns)

    def test_inferring_compatible_version(self) -> None:
        """Test that we can correctly infer the version a set is compatible with."""
        msdf10 = parse_sssom_table(f"{data_dir}/basic.tsv")

        # Nothing in that set requires 1.1
        self.assertEqual("1.0", msdf10.get_compatible_version())

        def _clone(msdf: MappingSetDataFrame) -> MappingSetDataFrame:
            return MappingSetDataFrame(df=msdf.df.copy(), metadata=msdf.metadata.copy())

        # Inject a 1.1-specific mapping set slot
        msdf11 = _clone(msdf10)
        msdf11.metadata[CARDINALITY_SCOPE] = "predicate_id"
        self.assertEqual("1.1", msdf11.get_compatible_version())

        # Inject a 1.1-specific mapping slot
        msdf11 = _clone(msdf10)
        msdf11.df[PREDICATE_TYPE] = "owl object property"
        self.assertEqual("1.1", msdf11.get_compatible_version())

        # Inject a 1.1-specific entity_type_enum value
        msdf11 = _clone(msdf10)
        msdf11.metadata[SUBJECT_TYPE] = "composed entity expression"
        self.assertEqual("1.1", msdf11.get_compatible_version())

        # Same, but on a single mapping record
        msdf11 = _clone(msdf10)
        msdf11.df[OBJECT_TYPE] = "owl class"
        msdf11.df.loc[2, OBJECT_TYPE] = "composed entity expression"
        self.assertEqual("1.1", msdf11.get_compatible_version())

        # Inject the 1.1-specific "0:0" cardinality value
        msdf11 = _clone(msdf10)
        msdf11.df[MAPPING_CARDINALITY] = "1:1"
        msdf11.df.loc[9, MAPPING_CARDINALITY] = "0:0"
        self.assertEqual("1.1", msdf11.get_compatible_version())

    def test_enforce_version(self) -> None:
        """Test that we can force a set to be compliant with a specific SSSOM version."""
        msdf11 = parse_sssom_table(f"{data_dir}/sssom11-extensions.sssom.tsv")

        # The test set contains non-standard slots, but they are
        # discarded by the parser (even those properly declared as
        # extensions!). To be able to test the "strict" enforcing mode,
        # we manually reintroduce the non-standard slots here.
        msdf11.metadata["ext_fooability_scale"] = 79
        msdf11.metadata["ext_undefined"] = "bar"
        msdf11.df["ext_fooable"] = True
        msdf11.df["ext_undefined"] = "bar"

        msdf10 = msdf11.enforce_version("1.0")
        # msdf11 should still have all its 1.1 slots since we are not
        # using inplace=True
        self.assertIn(MAPPING_SET_CONFIDENCE, msdf11.metadata)
        self.assertIn(MAPPING_TOOL_ID, msdf11.df.columns)
        self.assertIn("composed entity expression", msdf11.df[SUBJECT_TYPE].values)
        # But those slots should not be present in msdf10
        self.assertNotIn(MAPPING_SET_CONFIDENCE, msdf10.metadata)
        self.assertNotIn(MAPPING_TOOL_ID, msdf10.df.columns)
        self.assertNotIn("composed entity expression", msdf10.df[SUBJECT_TYPE].values)
        # Further confirm that msdf10 is 1.0-compliant
        self.assertEqual("1.0", msdf10.get_compatible_version())
        # Non-standard slots should all be preserved
        self.assertIn("ext_fooability_scale", msdf10.metadata)
        self.assertIn("ext_undefined", msdf10.metadata)
        self.assertIn("ext_fooable", msdf10.df.columns)
        self.assertIn("ext_undefined", msdf10.df.columns)

        msdf10 = msdf11.enforce_version("1.0", strict=True)
        self.assertEqual("1.0", msdf10.get_compatible_version())
        # Declared non-standard slots should still be there
        self.assertIn("ext_fooability_scale", msdf10.metadata)
        self.assertIn("ext_fooable", msdf10.df.columns)
        # But not undeclared ones
        self.assertNotIn("ext_undefined", msdf10.metadata)
        self.assertNotIn("ext_undefined", msdf10.df.columns)

        msdf11.enforce_version("1.0", inplace=True)
        # now msdf11 itself should be 1.0-compliant
        self.assertEqual("1.0", msdf11.get_compatible_version())
