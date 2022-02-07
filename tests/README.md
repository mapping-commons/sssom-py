# SSSOM Tests

The conversion framework for SSSOM testing is based on three components:
1. A config file ([test_config.yaml](test_config.yaml)) that lists all the test cases, from 
raw `filename` to metrics such as number of mappings.
2. A testcase class (`SSSOMTestCase`) in [test_data.py](test_data.py) that reads this config.
3. A framework ([test_conversion.py](test_conversion.py)) that dynamically runs all test cases and tests them against the metrics specified
in the config

At first, the rest results are a bit hard to read this way: if you have dynamic testing, it seems
any failures are at first a bit unintelligble, because you cannot immediately see what want wrong.
But if you use an IDE, you should be able to simply navigate to method an error was thrown to get
to the bottom of it.

## Test converters
1. to_dataframe
2. to_rdf_graph
3. to_owl_graph
4. to_dict

## Test parsers

1. from_tsv
   - basic.tsv
2. from_rdf
   - basic.ttl
3. from_alignment_form
   - oaei_ordo_hp.rdf
4. from_owl
5. from_json_ld
6. from_x

## Test writers

1. write_tsv
   - basic.tsv
2. write_owl
   - basic.tsv
3. write_rdf
   - basic.tsv
4. write_json_ld
   - basic.tsv
   