set -e

INPUT_FILE_1='tests/data/basic.tsv'
INPUT_FILE_2='tests/data/basic2.tsv'
INPUT_FILE_3='tests/data/basic3.tsv'
OUTPUT_DIR='tests/tmp/'
QUERY_1="SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence"

sssom parse --input $INPUT_FILE_1 --output $OUTPUT_DIR/parsed_basic.tsv --input-format tsv --curie-map-mode merged
sssom split --input $INPUT_FILE_1 --output-directory $OUTPUT_DIR

sssom convert --input $INPUT_FILE_1 --output $OUTPUT_DIR/converted_basic.tsv --output-format tsv
sssom validate --input $INPUT_FILE_1
sssom ptable --input $INPUT_FILE_1
sssom dedupe --input $INPUT_FILE_1 --output $OUTPUT_DIR/deduped_basic.tsv
sssom dosql -q "$QUERY_1" $INPUT_FILE_1 -o $OUTPUT_DIR/dosql_output.tsv
#sssom sparql 
sssom diff $INPUT_FILE_1 $INPUT_FILE_2 -o $OUTPUT_DIR/diff_output.tsv
sssom partition -d $OUTPUT_DIR $INPUT_FILE_1 $INPUT_FILE_2 
sssom cliquesummary -i $INPUT_FILE_1 -o $OUTPUT_DIR/cliquesummary_output.tsv
sssom crosstab -i $INPUT_FILE_1 -o $OUTPUT_DIR/crosstab_output.tsv
sssom correlations -i $INPUT_FILE_1 -o $OUTPUT_DIR/correlation_output.tsv
sssom merge $INPUT_FILE_1 $INPUT_FILE_2 $INPUT_FILE_3 -o $OUTPUT_DIR/merged_msdf.tsv

