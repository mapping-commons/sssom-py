from click.testing import CliRunner
from sssom.cli import convert
import unittest
from tests.test_data import ensure_test_dir_exists, DEFAULT_CONTEXT_PATH, get_all_test_cases


class SSSOMCLITestSuite(unittest.TestCase):

    def test_convert_cli(self):
        ensure_test_dir_exists()
        runner = CliRunner()
        test_cases = get_all_test_cases()
        for test in test_cases:
            result = run_convert(runner, test.filepath, test.get_out_file("tsv"), test.inputformat, "owl",
                                 DEFAULT_CONTEXT_PATH)
            self.assertTrue(result.exit_code == 0, f"The run of sssom convert failed with message {result.exception}")
        self.assertTrue(len(test_cases) > 2)


def run_convert(runner, input: str, output: str, format: str, to_format: str, context: str):
    return runner.invoke(convert, ['--input', input, '--output', output, '--format', format, '--to-format', to_format,
                                   '--context', context])
