"""Technical state-dictionary candidate matching, without semantic labeling."""

from stage1.state_mapping.core import (
    MAPPING_VERSION,
    STATE_MAPPING_SCHEMA,
    StateDefinition,
    StateDictionary,
    build_audit_table,
    load_state_dictionary,
    match_state,
)

__all__ = [
    "MAPPING_VERSION",
    "STATE_MAPPING_SCHEMA",
    "StateDefinition",
    "StateDictionary",
    "build_audit_table",
    "load_state_dictionary",
    "match_state",
]
