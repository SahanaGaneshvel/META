"""Tariff document -> TariffSpec with Muse Spark, checked by reproducing real bills.

The model gets the document as text (so the same reader also works on models without PDF input) and
must answer in the strict TariffSpec JSON schema. Two verifiers:
  - field accuracy against a hand-checked gold file, when one exists;
  - bill reproduction: the extracted spec and the gold spec bill the same real metered load for
    every month of 2016-2017; the worst monthly difference must be within 1%.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from ..meta.client import ModelClient
from ..tariff import TariffSpec, monthly_bills

SYSTEM = """You read electricity tariff documents for an EV-charging site operator.
Return the tariff as JSON matching the schema exactly. Rules:
- Use TOTAL rates (sum of all unbundled components such as commodity, distribution, public benefits).
- Hours are local clock hours on a 24-hour scale; a window is [start_hour, end_hour).
- An energy period that covers "all other hours" has an empty windows list.
- A demand charge that applies at all hours has an empty windows list.
- days: "weekdays" means Monday-Friday; set holidays_off_peak true if holidays are treated like weekends.
- demand_interval_minutes is the interval over which billed demand is averaged.
- A charge quoted per day becomes per month as rate x 365 / 12.
- If a period applies only in some months of a season, split the season so every window applies to all
  of its months.
Copy numbers exactly as printed; do not round."""


def document_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [f"--- page {i + 1} ---\n{page.extract_text()}" for i, page in enumerate(reader.pages)]
    return "\n".join(pages)


def tariff_schema() -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": "TariffSpec", "strict": True, "schema": TariffSpec.model_json_schema()}}


def extract(path: Path, client: ModelClient | None = None, reasoning_effort: str = "medium") -> tuple[TariffSpec, dict]:
    client = client or ModelClient("spark")
    response = client.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Tariff document ({path.name}):\n\n{document_text(path)}"}],
        response_format=tariff_schema(), reasoning_effort=reasoning_effort)
    content = response["choices"][0]["message"]["content"]
    return TariffSpec.model_validate_json(content), response


def _months(months: list[int]) -> str:
    return "months " + ",".join(str(m) for m in sorted(months))


def _hours(windows) -> str:
    """A charge is identified by when it applies, not by what the utility calls it."""
    if not windows:
        return "all other hours"
    return " + ".join(f"{w.days} {w.start_hour:g}-{w.end_hour:g}h" for w in sorted(
        windows, key=lambda w: (w.days, w.start_hour, w.end_hour)))


def _flatten(spec: TariffSpec) -> dict[str, object]:
    """Fields compared by meaning: a season is its months and a rate is the hours it applies to, so
    "Maximum Peak Demand Summer" and "peak demand" match when they cover the same hours."""
    out: dict[str, object] = {
        "demand_interval_minutes": spec.demand_interval_minutes,
        "holidays_off_peak": spec.holidays_off_peak,
        "customer_charge_usd_per_month": round(spec.customer_charge_usd_per_month, 5),
    }
    for season in spec.seasons:
        key = _months(season.months)
        out[f"{key}: season"] = True
        for rate in season.energy:  # a rate at the wrong hours has no match, so it counts as wrong
            out[f"{key}: energy at {_hours(rate.windows)}: $/kWh"] = round(rate.usd_per_kwh, 5)
        for charge in season.demand:
            hours = _hours(charge.windows).replace("all other hours", "all hours")
            out[f"{key}: demand at {hours}: $/kW"] = round(charge.usd_per_kw, 5)
    return out


def field_accuracy(extracted: TariffSpec, gold: TariffSpec) -> dict:
    got, want = _flatten(extracted), _flatten(gold)
    wrong = {k: {"gold": want[k], "extracted": got.get(k)} for k in want if got.get(k) != want[k]}
    return {"fields": len(want), "correct": len(want) - len(wrong), "accuracy": 1 - len(wrong) / len(want),
            "wrong": wrong}


def bill_check(extracted: TariffSpec, gold: TariffSpec, load_kw: pd.Series, tolerance: float = 0.01) -> dict:
    """Bill the same real metered load under both specs, month by month."""
    a = monthly_bills(load_kw, extracted)["total_usd"]
    b = monthly_bills(load_kw, gold)["total_usd"]
    error = ((a - b) / b).abs()
    return {"months": len(b), "worst_month": str(error.idxmax()), "worst_error": float(error.max()),
            "passed": bool(error.max() <= tolerance), "gold_total_usd": float(b.sum()),
            "extracted_total_usd": float(a.sum())}


def report(extracted: TariffSpec, gold: TariffSpec | None, load_kw: pd.Series) -> str:
    lines = [f"{extracted.utility} {extracted.schedule} (effective {extracted.effective})"]
    if gold is not None:
        acc = field_accuracy(extracted, gold)
        bills = bill_check(extracted, gold, load_kw)
        lines += [f"fields correct: {acc['correct']}/{acc['fields']}",
                  f"bill reproduction: worst month {bills['worst_month']} off by {bills['worst_error']:.3%} "
                  f"-> {'PASS' if bills['passed'] else 'FAIL'}"]
        lines += [f"  wrong: {k}: gold {v['gold']} vs {v['extracted']}" for k, v in acc["wrong"].items()]
    return "\n".join(lines)


def save(spec: TariffSpec, path: Path) -> None:
    path.write_text(json.dumps(spec.model_dump(), indent=2), encoding="utf-8")
