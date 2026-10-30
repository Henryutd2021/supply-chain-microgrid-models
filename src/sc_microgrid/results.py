"""Write solution summaries and tables to an output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Union

import pandas as pd


def to_jsonable(obj):
    """Convert summaries (tuple keys, numpy numbers, data frames) to JSON-compatible objects."""
    if isinstance(obj, Mapping):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    return obj


def keyed_table(table: Mapping, names) -> pd.DataFrame:
    """A data frame from a dict with tuple keys, e.g. {(site, tech): MW} -> columns site, tech, value."""
    rows = [dict(zip(names, k if isinstance(k, tuple) else (k,)), value=v) for k, v in table.items()]
    return pd.DataFrame(rows, columns=list(names) + ["value"])


def write_summary(summary: Mapping, out_dir: Union[str, Path], name: str = "summary.json") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(to_jsonable(summary), indent=1) + "\n")
    return path
