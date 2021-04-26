from click.testing import CliRunner
from sssom.cli import convert
import unittest
from tests.test_data import ensure_test_dir_exists, DEFAULT_CONTEXT_PATH, TEST_CASES


class SSSOMCLITestSuite(unittest.TestCase):

    def test_convert_cli(self):
        ensure_test_dir_exists()
        runner = CliRunner()
        for test in TEST_CASES:
            result = run_convert(runner, test.filepath, test.get_out_file("tsv"), test.inputformat, "owl",
                                 DEFAULT_CONTEXT_PATH)
            self.assertTrue(result.exit_code == 0)
        self.assertTrue(len(TEST_CASES) > 2)


def run_convert(runner,input: str, output: str, format: str, to_format: str, context: str):
    return runner.invoke(convert, ['--input', input, '--output', output,'--format', format, '--to-format',to_format, '--context', context])
