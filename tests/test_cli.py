"""Tests for the command line interface."""

import importlib.util
import os
import subprocess  # noqa
import unittest
from pathlib import Path
from typing import Any, Mapping, Optional, cast

from click.testing import CliRunner, Result
from rdflib import Graph, URIRef

from sssom.cli import (
    annotate,
    cliquesummary,
    convert,
    correlations,
    crosstab,
    dedupe,
    diff,
    dosql,
    filter,
    merge,
    parse,
    partition,
    ptable,
    reconcile_prefixes,
    remove,
    rewire,
    sort,
    split,
    validate,
)
from tests.constants import data_dir, test_out_dir
from tests.test_data import (
    RECON_YAML,
    SSSOMTestCase,
    get_all_test_cases,
    get_multiple_input_test_cases,
)


class SSSOMCLITestSuite(unittest.TestCase):
    """A test case for the dynamic CLI tests."""

    def test_cli_single_input(self) -> None:
        """Run all test cases on a single input file."""
        runner = CliRunner()
        # Initially returned 2 tsv and 1 rdf. The RDF failed test
        test_cases = get_all_test_cases()
        for test in test_cases:
            if test.inputformat == "alignment-api-xml":
                self.run_parse(runner, test)
            elif test.inputformat == "tsv":
                # These test only run on TSV inputs
                self.run_convert(runner, test)
                self.run_convert(runner, test, "ontoportal_json")
                self.run_convert(runner, test, "fhir_json")
                self.run_validate(runner, test)
                self.run_parse(runner, test)

                #####
                # This `if` condition needs to change (?)
                #####
                if test.filename == "basic.tsv":
                    self.run_split(runner, test)
                    self.run_ptable(runner, test)
                    self.run_dedupe(runner, test)
                    self.run_cliquesummary(runner, test)
                    self.run_crosstab(runner, test)
                    self.run_correlations(runner, test)
                    self.run_reconcile_prefix(runner, test)
                    if importlib.util.find_spec("pansql"):
                        self.run_dosql(runner, test)
                        self.run_filter(runner, test)
                    self.run_sort_rows_columns(runner, test)
                    self.run_annotate(runner, test)
                    self.run_remove(runner, test)

        self.assertTrue(len(test_cases) > 2)

    def test_cli_multiple_input(self) -> None:
        """Run test cases that require multiple files."""
        runner = CliRunner()
        test_cases = get_multiple_input_test_cases()
        self.run_diff(runner, test_cases)
        self.run_partition(runner, test_cases)
        self.run_merge(runner, test_cases)

        self.assertTrue(len(test_cases) >= 2)

    def test_validate_reports_errors_without_traceback(self) -> None:
        """Test that validation errors are reported as a clean CLI error."""
        text = (data_dir / "basic.tsv").read_text()
        bad_file = Path(test_out_dir) / "basic_non_uri_mapping_set_id.tsv"
        bad_file.write_text(
            text.replace(
                "#mapping_set_id: https://w3id.org/sssom/mapping/tests/data/basic.tsv",
                "#mapping_set_id: not_a_uri",
            )
        )
        result = CliRunner().invoke(validate, [bad_file.as_posix()])
        self.assertEqual(1, result.exit_code)
        self.assertIsInstance(result.exception, SystemExit)
        self.assertIn("Error: Your mapping set has 1 validation error:", result.output)
        self.assertIn("'not_a_uri' is not a 'uri' in /mapping_set_id", result.output)

    def test_validate_lists_all_errors(self) -> None:
        """Test that the CLI lists every validation error, not only the first few."""
        lines = (data_dir / "basic.tsv").read_text().splitlines()
        header_index = next(i for i, line in enumerate(lines) if not line.startswith("#"))
        confidence_index = lines[header_index].split("\t").index("confidence")
        first_row_index = header_index + 1
        rows = []
        for line in lines[first_row_index:]:
            row = line.split("\t")
            row[confidence_index] = "1.5"
            rows.append("\t".join(row))
        bad_file = Path(test_out_dir) / "basic_bad_confidence.tsv"
        bad_file.write_text("\n".join(lines[:first_row_index] + rows) + "\n")

        result = CliRunner().invoke(validate, [bad_file.as_posix()])
        self.assertEqual(1, result.exit_code)
        self.assertIn(f"Your mapping set has {len(rows)} validation errors:", result.output)
        self.assertNotIn("... and", result.output)
        for i in range(len(rows)):
            self.assertIn(f"in /mappings/{i}/confidence", result.output)

    def run_successful(self, result: Result, obj: Any) -> None:
        """Check the test result is successful."""
        if result.exit_code:
            raise RuntimeError(f"{obj} failed") from result.exception

    def run_convert(self, runner: CliRunner, test_case: SSSOMTestCase, fmt: str = "owl") -> Result:
        """Run the convert test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file(fmt),
            "--output-format",
            fmt,
        ]
        result = runner.invoke(convert, params)
        self.run_successful(result, test_case)
        return result

    def run_validate(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the validate test."""
        result = runner.invoke(validate, [test_case.filepath])
        self.run_successful(result, test_case)
        return result

    def run_parse(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the parse test."""
        if test_case.inputformat is None:
            raise TypeError

        params: list[str] = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
            "--input-format",
            test_case.inputformat,
            "--prefix-map-mode",
            "merged",
        ]
        if test_case.metadata_file:
            params.append("--metadata")
            params.append(data_dir / test_case.metadata_file)

        result: Result = runner.invoke(parse, params)
        self.run_successful(result, test_case)
        return result

    def run_split(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the split test."""
        result = runner.invoke(
            split, [test_case.filepath, "--output-directory", test_out_dir.as_posix()]
        )
        self.run_successful(result, test_case)
        return result

    # Added by H2

    def run_ptable(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the ptable test."""
        params = [test_case.filepath, "--output", test_case.get_out_file("ptable.tsv")]
        result = runner.invoke(ptable, params)
        self.run_successful(result, test_case)
        return result

    def run_dedupe(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the deduplication test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(dedupe, params)
        self.run_successful(result, test_case)
        return result

    # TODO: dosql and sparql
    """def run_dosql(self, runner, test_case: SSSOMTestCase):
        params = ['--query', test_case._query_tuple, '']"""

    """def run_sparql(self, runner, test_case: SSSOMTestCase):
        prams = []"""

    def run_diff(self, runner: CliRunner, test_cases: Mapping[str, SSSOMTestCase]) -> Result:
        """Run the diff test."""
        params = []
        out_file = None
        for t in test_cases.values():
            params.append(t.filepath)
            out_file = t
        if out_file:
            params.extend(["--output", out_file.get_out_file("tsv")])
            result = runner.invoke(diff, params)
            self.run_successful(result, out_file)
            return result
        else:
            self.fail("No test to run.")

    def run_partition(self, runner: CliRunner, test_cases: Mapping[str, SSSOMTestCase]) -> Result:
        """Run the partition test."""
        params = []
        primary_test_case: Optional[SSSOMTestCase] = None
        for t in test_cases.values():
            if not primary_test_case:
                primary_test_case = t
            params.append(t.filepath)
        primary_test_case = cast(SSSOMTestCase, primary_test_case)
        name = Path(primary_test_case.filepath).stem
        directory = test_out_dir.joinpath(name)
        directory.mkdir(exist_ok=True, parents=True)
        params.extend(["--output-directory", directory.as_posix()])
        result = runner.invoke(partition, params)
        self.run_successful(result, primary_test_case)
        return result

    def run_cliquesummary(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the clique summary test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(cliquesummary, params)
        self.run_successful(result, test_case)
        return result

    def run_crosstab(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the cross-tabulation test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(crosstab, params)
        self.run_successful(result, test_case)
        return result

    def run_correlations(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the correlations test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(correlations, params)
        self.run_successful(result, test_case)
        return result

    def run_merge(self, runner: CliRunner, test_cases: Mapping[str, SSSOMTestCase]) -> Result:
        """Run the merge test."""
        params = []
        out_file = None
        for t in test_cases.values():
            params.append(t.filepath)
            out_file = t
        if out_file:
            params.extend(["--output", out_file.get_out_file("tsv")])

        result = runner.invoke(merge, params)
        self.run_successful(result, test_cases)
        return result

    def run_reconcile_prefix(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the merge test with reconcile prefixes."""
        out_file = os.path.join(test_out_dir, "reconciled_prefix.tsv")
        result = runner.invoke(
            reconcile_prefixes,
            [
                test_case.filepath,
                "--output",
                out_file,
                "--reconcile-prefix-file",
                RECON_YAML,
            ],
        )
        self.run_successful(result, test_case)
        return result

    def run_dosql(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Test a simple dosql command."""
        out_file = os.path.join(test_out_dir, "dosql_test.tsv")
        result = runner.invoke(
            dosql,
            [
                "-Q",
                "SELECT * FROM df WHERE subject_label = 'heart'",
                test_case.filepath,
                "-o",
                os.path.join(test_out_dir, out_file),
            ],
        )
        self.run_successful(result, test_case)
        return result

    def run_sort_rows_columns(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Test sorting of DataFrame columns."""
        out_file = os.path.join(test_out_dir, "sort_column_test.tsv")
        in_file = test_case.filepath.replace("basic", "basic6")
        result = runner.invoke(
            sort,
            [
                in_file,
                "-o",
                os.path.join(test_out_dir, out_file),
                "-k",
                "true",
                "-r",
                "true",
            ],
        )
        self.run_successful(result, test_case)
        return result

    def run_filter(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Test filtering of DataFrame columns."""
        out_file = os.path.join(test_out_dir, "filter_test.tsv")
        in_file = test_case.filepath
        result = runner.invoke(
            filter,
            [
                in_file,
                "-o",
                os.path.join(test_out_dir, out_file),
                "--subject_id",
                "x:%",
                "--subject_id",
                "y:%",
                "--object_id",
                "y:%",
                "--object_id",
                "z:%",
            ],
        )
        self.run_successful(result, test_case)
        return result

    def run_annotate(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Test annotation of mapping set metadata."""
        out_file = os.path.join(test_out_dir, "test_annotate.tsv")
        in_file = test_case.filepath
        result = runner.invoke(
            annotate,
            [
                in_file,
                "-o",
                os.path.join(test_out_dir, out_file),
                "--mapping_set_id",
                "http://w3id.org/my/mapping.sssom.tsv",
                "--mapping_set_version",
                "2021-01-01",
            ],
        )
        self.run_successful(result, test_case)
        return result

    def run_remove(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Test removal of mappings."""
        out_file = os.path.join(test_out_dir, "remove_map_test.tsv")
        in_file = test_case.filepath
        rm_file = os.path.join(data_dir, "basic3.tsv")
        result = runner.invoke(
            remove,
            [
                in_file,
                "-o",
                os.path.join(test_out_dir, out_file),
                "--remove-map",
                rm_file,
            ],
        )
        self.run_successful(result, test_case)
        return result

    @unittest.skip("this test doesn't actually test anything, just runs help")
    def test_convert_cli(self) -> None:
        """Test conversion of SSSOM tsv to OWL format when multivalued metadata items are present."""
        test_sssom = data_dir / "test_inject_metadata_msdf.tsv"
        args: list[str] = [
            "sssom",
            "convert",
            test_sssom.as_posix(),
            "--output-format",
            "owl",
        ]
        result = subprocess.check_output(args, shell=True)  # noqa

    def test_rewire_repeated_predicates(self) -> None:
        """Test rewiring a graph by predicates given in a repeated option, with precedence."""
        runner = CliRunner()
        out_file = os.path.join(test_out_dir, "rewire-predicates-exact-close.ttl")
        result = runner.invoke(
            rewire,
            [
                os.path.join(data_dir, "rewire-predicates.ttl"),
                "-m",
                os.path.join(data_dir, "rewire-predicates.tsv"),
                "--predicates",
                "skos:exactMatch",
                "--predicates",
                "skos:closeMatch",
                "--precedence",
                "ibm-risk-atlas",
                "-o",
                out_file,
            ],
        )
        self.run_successful(result, "rewire --predicates, repeated")
        nodes = set(Graph().parse(out_file, format="turtle").all_nodes())
        ibm_risk_atlas = "https://www.ibm.com/docs/en/watsonx/saas?topic="
        self.assertIn(URIRef(ibm_risk_atlas + "atlas-output-bias"), nodes)
        self.assertIn(URIRef(ibm_risk_atlas + "atlas-toxic-output"), nodes)
        self.assertNotIn(URIRef(ibm_risk_atlas + "atlas-harmful-output"), nodes)
        self.assertIn(URIRef("https://taxonomy.eticas.ai/eticas-sentiment-fairness"), nodes)

    def test_rewire_predicates(self) -> None:
        """Test rewiring a graph by a comma-separated list of predicates in place of the default."""
        runner = CliRunner()
        out_file = os.path.join(test_out_dir, "rewire-predicates-exact-narrow.ttl")
        result = runner.invoke(
            rewire,
            [
                os.path.join(data_dir, "rewire-predicates.ttl"),
                "-m",
                os.path.join(data_dir, "rewire-predicates.tsv"),
                "--predicates",
                "skos:exactMatch,skos:narrowMatch",
                "-o",
                out_file,
            ],
        )
        self.run_successful(result, "rewire --predicates")
        nodes = set(Graph().parse(out_file, format="turtle").all_nodes())
        ibm_risk_atlas = "https://www.ibm.com/docs/en/watsonx/saas?topic="
        self.assertIn(URIRef(ibm_risk_atlas + "atlas-output-bias"), nodes)
        self.assertIn(URIRef(ibm_risk_atlas + "atlas-toxic-output"), nodes)
        self.assertIn(URIRef("https://airisk.mit.edu/mit-ai-risk-subdomain-1.1"), nodes)
        self.assertNotIn(URIRef("https://taxonomy.eticas.ai/eticas-sentiment-fairness"), nodes)

    def test_rewire_strict(self) -> None:
        """Test that rewire fails with --strict when no mapping is left to rewire by."""
        runner = CliRunner()
        args = [
            os.path.join(data_dir, "cob.owl"),
            "-I",
            "xml",
            "-m",
            os.path.join(data_dir, "mondo-nando.sssom.tsv"),
            "-o",
            os.path.join(test_out_dir, "rewire-strict.ttl"),
        ]
        self.run_successful(runner.invoke(rewire, args), "rewire without --strict")
        result = runner.invoke(rewire, [*args, "--strict"])
        self.assertEqual(1, result.exit_code)
        self.assertIn("No mapping to rewire by", result.output)

    def test_rewire_ambiguous(self) -> None:
        """Test that rewire reports an ambiguous mapping as a one-line error, not a traceback."""
        runner = CliRunner()
        args = [
            os.path.join(data_dir, "cob.owl"),
            "-I",
            "xml",
            "-m",
            os.path.join(data_dir, "basic.tsv"),
            "-o",
            os.path.join(test_out_dir, "rewire-ambiguous.ttl"),
        ]
        result = runner.invoke(rewire, args)
        self.assertEqual(1, result.exit_code)
        # Only a ClickException shows in the output, as "Error: <message>".
        self.assertIn("Error: Ambiguous", result.output)
