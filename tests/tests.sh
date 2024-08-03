set -e

INPUT_FILE_1='tests/data/basic.tsv'
INPUT_FILE_2='tests/data/basic2.tsv'
INPUT_FILE_3='tests/data/basic3.tsv'
INPUT_URL_1='https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic.tsv'
INPUT_URL_2='https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic2.tsv'
INPUT_URL_3='https://raw.githubusercontent.com/mapping-commons/sssom-py/master/tests/data/basic3.tsv'
OUTPUT_DIR='tests/tmp/'
QUERY_1="SELECT * FROM df1 WHERE confidence>0.5 ORDER BY confidence"

mkdir -p $OUTPUT_DIR

echo "Testing parse..."
sssom parse $INPUT_FILE_1 --output $OUTPUT_DIR/parsed_basic_file.tsv --input-format tsv --prefix-map-mode merged
sssom parse $INPUT_URL_1 --output $OUTPUT_DIR/parsed_basic_url.tsv --input-format tsv --prefix-map-mode merged

echo "Testing split..."
sssom split $INPUT_FILE_1 --output-directory $OUTPUT_DIR
sssom split $INPUT_URL_1 --output-directory $OUTPUT_DIR

echo "Testing convert..."
sssom convert $INPUT_FILE_1 --output $OUTPUT_DIR/converted_basic_file.tsv --output-format tsv
sssom convert $INPUT_URL_1 --output $OUTPUT_DIR/converted_basic_url.tsv --output-format tsv

for fmt in tsv json owl rdf
do
  sssom convert $INPUT_FILE_1 --output $OUTPUT_DIR/converted_basic_file.$fmt --output-format $fmt
  sssom convert $INPUT_URL_1 --output $OUTPUT_DIR/converted_basic_url.$fmt --output-format $fmt
done

echo "Testing validate..."
sssom validate $INPUT_FILE_1
sssom validate $INPUT_URL_1

echo "Testing ptable..."
sssom ptable $INPUT_FILE_1 > $OUTPUT_DIR/basic_file_ptable.tsv
sssom ptable $INPUT_URL_1 > $OUTPUT_DIR/basic_url_ptable.tsv

echo "Testing dedupe..."
sssom dedupe $INPUT_FILE_1 --output $OUTPUT_DIR/deduped_basic_file.tsv
sssom dedupe $INPUT_URL_1 --output $OUTPUT_DIR/deduped_basic_url.tsv

# sssom dosql -q "$QUERY_1" $INPUT_FILE_1 -o $OUTPUT_DIR/dosql_output_file.tsv
# sssom dosql -q "$QUERY_1" $INPUT_URL_1 -o $OUTPUT_DIR/dosql_output_url.tsv

# sssom sparql 

echo "Testing diff..."
sssom diff $INPUT_FILE_1 $INPUT_FILE_2 -o $OUTPUT_DIR/diff_output_file.tsv
sssom diff $INPUT_URL_1 $INPUT_URL_2 -o $OUTPUT_DIR/diff_output_url.tsv

echo "Testing partition..."
sssom partition -d $OUTPUT_DIR $INPUT_FILE_1 $INPUT_FILE_2 
sssom partition -d $OUTPUT_DIR $INPUT_URL_1 $INPUT_URL_2

echo "Testing cliquesummary..."
sssom cliquesummary $INPUT_FILE_1 -o $OUTPUT_DIR/cliquesummary_output_file.tsv
sssom cliquesummary $INPUT_URL_1 -o $OUTPUT_DIR/cliquesummary_output_url.tsv

echo "Testing crosstab..."
sssom crosstab $INPUT_FILE_1 -o $OUTPUT_DIR/crosstab_output_file.tsv
sssom crosstab $INPUT_URL_1 -o $OUTPUT_DIR/crosstab_output_url.tsv

echo "Testing correlations..."
sssom correlations $INPUT_FILE_1 -o $OUTPUT_DIR/correlation_output_file.tsv
sssom correlations $INPUT_URL_1 -o $OUTPUT_DIR/correlation_output_url.tsv

echo "Testing merge..."
sssom merge $INPUT_FILE_1 $INPUT_FILE_2 $INPUT_FILE_3 -o $OUTPUT_DIR/merged_msdf_file.tsv
sssom merge $INPUT_URL_1 $INPUT_URL_2 $INPUT_URL_3 -o $OUTPUT_DIR/merged_msdf_url.tsv