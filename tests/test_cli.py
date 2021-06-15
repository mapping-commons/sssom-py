from click.testing import CliRunner
from sssom.cli import convert, validate, split, parse
import unittest
from tests.test_data import ensure_test_dir_exists, get_all_test_cases, SSSOMTestCase, test_out_dir


class SSSOMCLITestSuite(unittest.TestCase):

    def test_convert_cli(self):
        ensure_test_dir_exists()
        runner = CliRunner()
        test_cases = get_all_test_cases()
        for test in test_cases:
            print(test.filepath)
            test: SSSOMTestCase
            self.run_convert(runner, test)
            if test.inputformat == "tsv":
                # These test only run on TSV inputs
                self.run_validate(runner, test)
                self.run_parse(runner, test)
                self.run_split(runner, test)
        self.assertTrue(len(test_cases) > 2)

    def run_successful(self, result):
        # self.assertTrue(result.exit_code == 0, f"Run failed with message {result.exception}")
        print(result.exit_code)

    def run_convert(self, runner, test_case: SSSOMTestCase):
        result = runner.invoke(convert, ['--input', test_case.filepath, '--output', test_case.get_out_file("tsv"),
                                         '--output-format', 'owl'])
        self.run_successful(result)
        return result

    def run_validate(self, runner, test_case: SSSOMTestCase):
        result = runner.invoke(validate, ['--input', test_case.filepath])
        self.run_successful(result)
        return result

    def run_parse(self, runner, test_case: SSSOMTestCase):
        params = ['--input', test_case.filepath, '--output', test_case.get_out_file("tsv"), '--input-format',
                  test_case.inputformat,
                  '--curie-map-mode', "merged"]
        if test_case.metadata_file:
            params.append('--metadata')
            params.append(test_case.metadata_file)

        result = runner.invoke(parse, params)
        self.run_successful(result)
        return result

    def run_split(self, runner, test_case: SSSOMTestCase):
        result = runner.invoke(split, ['--input', test_case.filepath, '--output-directory', test_out_dir])
        self.run_successful(result)
        return result
