from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    kind: str
    default: object
    choices: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class TrainingSchema:
    feature_columns: Tuple[str, ...]
    columns: Tuple[ColumnSchema, ...]
    defaults: Dict[str, object]


def _is_missing(v: Optional[str]) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"na", "nan", "none", "null"}


def _try_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def infer_training_schema(
    csv_path: str | Path,
    *,
    exclude_columns: Optional[Set[str]] = None,
    max_rows: int = 200,
) -> TrainingSchema:
    p = Path(csv_path)
    exclude = set(exclude_columns or set())
    exclude |= {
        "",
        "id",
        "case_id",
        "patient_id",
        "slide_id",
        "survival",
        "time",
        "os",
        "dfs",
        "censorship",
        "event",
        "status",
        "split",
        "nomogram_score",
        "survival_1",
        "survival_2",
        "survival_3",
    }

    with p.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return TrainingSchema(feature_columns=tuple(), columns=tuple(), defaults={})

        fieldnames = [c for c in reader.fieldnames if c is not None]
        feature_columns = [c for c in fieldnames if c not in exclude]

        numeric_ok = {c: 0 for c in feature_columns}
        numeric_fail = {c: 0 for c in feature_columns}
        numeric_sum = {c: 0.0 for c in feature_columns}
        numeric_cnt = {c: 0 for c in feature_columns}
        cat_counts: Dict[str, Dict[str, int]] = {c: {} for c in feature_columns}

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            for c in feature_columns:
                raw = row.get(c)
                if _is_missing(raw):
                    continue
                s = str(raw).strip()
                fv = _try_float(s)
                if fv is None:
                    numeric_fail[c] += 1
                    cat_counts[c][s] = cat_counts[c].get(s, 0) + 1
                else:
                    numeric_ok[c] += 1
                    numeric_sum[c] += fv
                    numeric_cnt[c] += 1

        columns: List[ColumnSchema] = []
        defaults: Dict[str, object] = {}

        for c in feature_columns:
            ok = numeric_ok[c]
            fail = numeric_fail[c]
            if ok > 0 and fail == 0:
                mean = numeric_sum[c] / numeric_cnt[c] if numeric_cnt[c] else 0.0
                if float(mean).is_integer():
                    default: object = int(mean)
                else:
                    default = float(mean)
                schema = ColumnSchema(name=c, kind="numeric", default=default, choices=None)
            else:
                counts = cat_counts[c]
                if counts:
                    choices = tuple(sorted(counts.keys()))
                    default = max(counts.items(), key=lambda kv: kv[1])[0]
                else:
                    choices = tuple()
                    default = ""
                schema = ColumnSchema(name=c, kind="categorical", default=default, choices=choices)

            columns.append(schema)
            defaults[c] = schema.default

    return TrainingSchema(
        feature_columns=tuple(feature_columns),
        columns=tuple(columns),
        defaults=defaults,
    )


def coerce_patient_row(
    schema: TrainingSchema,
    values: Dict[str, object],
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for col in schema.columns:
        v = values.get(col.name, col.default)
        if col.kind == "numeric":
            if v is None or _is_missing(str(v)):
                out[col.name] = col.default
            else:
                if isinstance(v, (int, float)):
                    out[col.name] = float(v)
                else:
                    out[col.name] = float(str(v).strip())
        else:
            out[col.name] = "" if v is None else str(v)
    return out
