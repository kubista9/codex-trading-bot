from __future__ import annotations

from typing import Any


def resolve_feature_columns(config: dict[str, Any], regime_name: str) -> list[str]:
    """Resolve configured feature columns for a named feature regime."""
    regimes = config.get("feature_regimes", {})
    groups = config.get("groups", {})
    if regime_name not in regimes:
        msg = f"Unknown feature regime: {regime_name}"
        raise ValueError(msg)

    columns: list[str] = []
    for group_name in regimes[regime_name].get("groups", []):
        group = groups.get(group_name)
        if group is None:
            msg = f"Feature regime {regime_name} references unknown group {group_name}"
            raise ValueError(msg)
        columns.extend(group.get("columns", []))

    return list(dict.fromkeys(columns))
