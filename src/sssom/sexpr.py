"""Generate canonical s-expressions and mapping hashes."""

import hashlib
import re
import unittest

import zbase32

from sssom import Mapping
from sssom.constants import _get_sssom_schema_object

__all__ = [
    "get_mapping_hash",
]

def get_mapping_hash(x: Mapping) -> str:
    """Hash the mapping by converting to canonical s-expression, sha256 hashing, then zbase32 encoding."""
    s = hashlib.sha256()
    s.update(to_sexpr(x).encode("utf-8"))
    dig = s.digest()
    return zbase32.encode(dig)


SKIP_SLOTS = {"record_id", "mapping_cardinality"}


def to_sexpr(x: Mapping) -> str:
    # todo get canonical order
    rv = "(7:mapping("
    for slot in _get_sssom_schema_object().slots:
        if slot in SKIP_SLOTS:
            continue
        value = getattr(x, slot, None)
        if not value:
            continue
        elif isinstance(value, str):
            rv += f"({len(slot)}:{slot}{len(value)}:{value})"
        elif isinstance(value, float):
            raise NotImplementedError
        elif isinstance(value, list):
            rv += f"({len(slot)}:{slot}("
            for v in value:
                rv += f"{len(v)}:{v}"
            rv += "))"
    return rv + "))"


class TestSExpressions(unittest.TestCase):
    def test_big_example(self) -> None:
        """"""
        s = """
        (7:mapping(
           (10:subject_id44:http://purl.obolibrary.org/obo/FBbt_00001234)
           (12:predicate_id46:http://www.w3.org/2004/02/skos/core#exactMatch)
           (9:object_id45:http://purl.obolibrary.org/obo/UBERON_0005678)
           (21:mapping_justification51:https://w3id.org/semapv/vocab/ManualMappingCuration)
           (10:creator_id(
                          37:https://orcid.org/0000-0000-1234-5678
                          37:https://orcid.org/0000-0000-5678-1234
            ))
        ))
        """
        x = Mapping(
            subject_id="http://purl.obolibrary.org/obo/FBbt_00001234",
            predicate_id="http://www.w3.org/2004/02/skos/core#exactMatch",
            object_id="http://purl.obolibrary.org/obo/UBERON_0005678",
            mapping_justification="https://w3id.org/semapv/vocab/ManualMappingCuration",
            creator_id=[
                "https://orcid.org/0000-0000-1234-5678",
                "https://orcid.org/0000-0000-5678-1234",
            ],
        )
        self.assertEqual(re.sub("\s", "", s), to_sexpr(x))
        self.assertEqual("hq6bs14aptzepwgk6pw7j8ysnft6riqrw7har84rtk8r9xmbcwty", get_mapping_hash(x))
