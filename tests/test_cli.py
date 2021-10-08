"""Tests for the command line interface."""

import unittest
from typing import Mapping

from click.testing import CliRunner, Result

from sssom.cli import (
    cliquesummary,
    convert,
    correlations,
    crosstab,
    dedupe,
    diff,
    parse,
    partition,
    ptable,
    split,
    validate,
)
from tests.test_data import (
    SSSOMTestCase,
    get_all_test_cases,
    get_multiple_input_test_cases,
    test_out_dir,
)


class SSSOMCLITestSuite(unittest.TestCase):
    """A test case for the dynamic CLI tests."""

    def test_cli_single_input(self):
        """Run all test cases on a single input file."""
        runner = CliRunner()
        # Initially returned 2 tsv and 1 rdf. The RDF failed test
        test_cases = get_all_test_cases()
        for test in test_cases:
            self.run_convert(runner, test)
            if test.inputformat == "tsv":
                # These test only run on TSV inputs
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
        self.assertTrue(len(test_cases) > 2)

    def test_cli_multiple_input(self):
        """Run test cases that require multiple files."""
        runner = CliRunner()
        test_cases = get_multiple_input_test_cases()
        self.run_diff(runner, test_cases)
        self.run_partition(runner, test_cases)

        self.assertTrue(len(test_cases) >= 2)

    def run_successful(self, result: Result, test_case: SSSOMTestCase) -> None:
        """Check the test result is successful."""
        # self.assertTrue(result.exit_code == 0, f"Run failed with message {result.exception}")
        self.assertEqual(
            result.exit_code, 0, f"{test_case} did not as expected: {result.exception}"
        )

    def run_convert(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the convert test."""
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
            "--output-format",
            "owl",
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
        params = [
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
            params.append(test_case.metadata_file)

        result = runner.invoke(parse, params)
        self.run_successful(result, test_case)
        return result

    def run_split(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the split test."""
        result = runner.invoke(
            split, [test_case.filepath, "--output-directory", test_out_dir]
        )
        self.run_successful(result, test_case)
        return result

    # Added by H2

    def run_ptable(self, runner: CliRunner, test_case: SSSOMTestCase) -> Result:
        """Run the ptable test."""
        params = [test_case.filepath]
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

    def run_diff(
        self, runner: CliRunner, test_cases: Mapping[str, SSSOMTestCase]
    ) -> Result:
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

    def run_partition(
        self, runner: CliRunner, test_cases: Mapping[str, SSSOMTestCase]
    ) -> Result:
        """Run the partition test."""
        params = []
        primary_test_case = None
        for t in test_cases.values():
            if not primary_test_case:
                primary_test_case = t
            params.append(t.filepath)
        params.extend(["--output-directory", test_out_dir])
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
