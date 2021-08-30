import unittest

from click.testing import CliRunner
from typing import Dict

from sssom.cli import (
    cliquesummary,
    convert,
    correlations,
    crosstab,
    dedupe,
    diff,
    partition,
    ptable,
    validate,
    split,
    parse,
)
from tests.test_data import (
    ensure_test_dir_exists,
    get_all_test_cases,
    SSSOMTestCase,
    test_out_dir,
    get_multiple_input_test_cases,
)


class SSSOMCLITestSuite(unittest.TestCase):
    def test_cli_single_input(self):
        ensure_test_dir_exists()
        runner = CliRunner()
        test_cases = (
            get_all_test_cases()
        )  # Initially returned 2 tsv and 1 rdf. The RDF failed test
        for test in test_cases:
            test: SSSOMTestCase
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
        ensure_test_dir_exists()
        runner = CliRunner()
        test_cases = get_multiple_input_test_cases()
        self.run_diff(runner, test_cases)
        self.run_partition(runner, test_cases)

        self.assertTrue(len(test_cases) >= 2)

    def run_successful(self, result, test_case: SSSOMTestCase):
        # self.assertTrue(result.exit_code == 0, f"Run failed with message {result.exception}")
        self.assertEqual(result.exit_code, 0, f"{test_case} did not as expected: {result.exception}")

    def run_convert(self, runner, test_case: SSSOMTestCase):
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

    def run_validate(self, runner, test_case: SSSOMTestCase):
        result = runner.invoke(validate, [test_case.filepath])
        self.run_successful(result, test_case)
        return result

    def run_parse(self, runner, test_case: SSSOMTestCase):
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
            "--input-format",
            test_case.inputformat,
            "--curie-map-mode",
            "merged",
        ]
        if test_case.metadata_file:
            params.append("--metadata")
            params.append(test_case.metadata_file)

        result = runner.invoke(parse, params)
        self.run_successful(result, test_case)
        return result

    def run_split(self, runner, test_case: SSSOMTestCase):
        result = runner.invoke(
            split, [test_case.filepath, "--output-directory", test_out_dir]
        )
        self.run_successful(result, test_case)
        return result

    # Added by H2

    def run_ptable(self, runner, test_case: SSSOMTestCase):
        params = [test_case.filepath]
        result = runner.invoke(ptable, params)
        self.run_successful(result, test_case)
        return result

    def run_dedupe(self, runner, test_case: SSSOMTestCase):
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

    def run_diff(self, runner, test_cases: Dict[str, SSSOMTestCase]):
        params = []
        out_file = None
        for tid in test_cases:
            t = test_cases[tid]
            t: SSSOMTestCase
            params.append(t.filepath)
            out_file = t
        if out_file:
            params.extend(["--output", out_file.get_out_file("tsv")])
            result = runner.invoke(diff, params)
            self.run_successful(result, out_file)
            return result
        else:
            self.fail("No test to run.")

    def run_partition(self, runner, test_cases: dict):
        params = []
        primary_test_case = None
        for tid in test_cases:
            t = test_cases[tid]
            if not primary_test_case:
                primary_test_case = t
            t: SSSOMTestCase
            params.append(t.filepath)
        params.extend(["--output-directory", test_out_dir])
        result = runner.invoke(partition, params)
        self.run_successful(result, primary_test_case)
        return result

    def run_cliquesummary(self, runner, test_case: SSSOMTestCase):
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(cliquesummary, params)
        self.run_successful(result, test_case)
        return result

    def run_crosstab(self, runner, test_case: SSSOMTestCase):
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(crosstab, params)
        self.run_successful(result, test_case)
        return result

    def run_correlations(self, runner, test_case: SSSOMTestCase):
        params = [
            test_case.filepath,
            "--output",
            test_case.get_out_file("tsv"),
        ]
        result = runner.invoke(correlations, params)
        self.run_successful(result, test_case)
        return result
