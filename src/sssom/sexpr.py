"""Generate canonical s-expressions and mapping hashes."""

import hashlib

import curies
import zbase32

from sssom import Mapping
from sssom.constants import _get_sssom_schema_object

__all__ = [
    "get_mapping_hash",
]


def get_mapping_hash(mapping: Mapping, converter: curies.Converter) -> str:
    """Hash the mapping by converting to canonical s-expression, sha256 hashing, then zbase32 encoding."""
    s = hashlib.sha256()
    s.update(to_sexpr(mapping, converter).encode("utf-8"))
    dig = s.digest()
    return zbase32.encode(dig)


SKIP_SLOTS = {"record_id", "mapping_cardinality"}


def _should_expand(slot: str) -> bool:
    return True


def to_sexpr(x: Mapping, converter: curies.Converter) -> str:
    # todo get canonical order
    rv = "(7:mapping("
    for slot in _get_sssom_schema_object().slots:
        if slot in SKIP_SLOTS:
            continue
        value = getattr(x, slot, None)
        if not value:
            continue
        elif isinstance(value, str):
            if _should_expand(slot):
                value = converter.expand_or_standardize(value, strict=True)
            # TODO check if it's an entity reference and should be expanded
            rv += f"({len(slot)}:{slot}{len(value)}:{value})"
        elif isinstance(value, float):
            raise NotImplementedError
        elif isinstance(value, list):
            rv += f"({len(slot)}:{slot}("
            for v in value:
                if _should_expand(slot):
                    v = converter.expand_or_standardize(v, strict=True)
                rv += f"{len(v)}:{v}"
            rv += "))"
    return rv + "))"
