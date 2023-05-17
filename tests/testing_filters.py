from sssom.constants import OBJECT_ID, SUBJECT_ID
from sssom.parsers import parse_sssom_table
from sssom.util import (
    filter_prefixes,
)
from tests.constants import data_dir

msdf = parse_sssom_table(f"tests/data/basic.tsv")
features = [SUBJECT_ID, OBJECT_ID]

prefix_filter_list = ["x", "y"]
# prefix_filter_list = ["x"]
original_msdf = msdf
filtered_df = filter_prefixes(
    original_msdf.df, prefix_filter_list, features
)
print(filtered_df)