set -e

sssom parse --input /Users/matentzn/ws/sssom-py/tests/data/basic.tsv --output /Users/matentzn/ws/sssom-py/tests/tmp/basic.tsv.tsv --input-format tsv --curie-map-mode merged
sssom split --input /Users/matentzn/ws/sssom-py/tests/data/basic.tsv --output-directory /Users/matentzn/ws/sssom-py/tests/tmp/
