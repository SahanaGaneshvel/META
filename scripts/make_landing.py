"""The project's landing page (docs/index.html, served by GitHub Pages), with every number and the chart
read from data/results/ and the real heat-wave day.

    python scripts/make_landing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from headroom.capacity import hosting_capacity  # noqa: E402
from headroom.config import DATA, ROOT  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

REPO = "https://github.com/SahanaGaneshvel/META"
VIDEO = "VIDEO_LINK"  # filled in once the video is uploaded


def numbers() -> dict:
    final = pd.read_csv(DATA / "results" / "final_summary.csv", index_col="controller")
    cap = hosting_capacity(pd.read_csv(DATA / "results" / "capacity_summary_test.csv"))
    p = pd.read_csv(DATA / "results" / "portfolio_summary.csv")
    both = p[p["chargers_standard_smart_charging"] > 0]
    spec = load_gold("e-4-tou_2026")
    summer = next(s for s in spec.seasons if 7 in s.months)
    cs = pd.read_csv(DATA / "results" / "capacity_summary_test.csv").set_index(["controller", "n_ports"])
    return {
        "extra": cs.loc[("headroom", 36), "added_demand_usd_mean"] - cs.loc[("headroom", 24), "added_demand_usd_mean"],
        "served36": cs.loc[("headroom", 36), "energy_served_mean"],
        "viol36": int(cs.loc[("headroom", 36), "service_violations"]),
        "spike": 50 * sum(d.usd_per_kw for d in summer.demand),
        "energy": 50 * 0.25 * max(r.usd_per_kwh for r in summer.energy),
        "saving": 1 - final.loc["headroom", "added_demand_usd"] / final.loc["llf", "added_demand_usd"],
        "served": final.loc["headroom", "energy_served"],
        "violations": int(final.loc["headroom", "service_violations"]),
        "with": int(cap["headroom"]), "without": int(cap["llf"]),
        "buildings": len(p), "sites": int(p["site_id"].nunique()),
        "more": (both["extra_chargers"] > 0).mean(),
        "median": (both["chargers_headroom"] / both["chargers_standard_smart_charging"] - 1).median(),
    }


def chart() -> dict:
    """SVG paths for the real heat-wave day: the building, and the building plus Headroom's charging."""
    from make_cards import heat_wave_day

    t, building, ev, _ = heat_wave_day()
    hours = np.asarray(t.hour + t.minute / 60)
    x = (hours - hours.min()) / (hours.max() - hours.min()) * 600
    y = lambda kw: 230 - (kw - 450) / (1000 - 450) * 210  # noqa: E731 - 450-1000 kW on a 230 px axis

    def area(top: np.ndarray, bottom: np.ndarray) -> str:
        pts = [f"{a:.1f},{y(b):.1f}" for a, b in zip(x, top)] + [f"{a:.1f},{y(b):.1f}" for a, b in zip(x[::-1], bottom[::-1])]
        return "M" + " L".join(pts) + " Z"

    floor = np.full_like(building, 450.0)
    return {"building": area(building, floor), "ev": area(building + ev, building), "limit": f"{y(1000):.1f}",
            "peak_x": f"{(16 - hours.min()) / (hours.max() - hours.min()) * 600:.1f}",
            "peak_w": f"{5 / (hours.max() - hours.min()) * 600:.1f}"}


def main() -> None:
    n, c = numbers(), chart()
    html = (Path(__file__).with_name("landing_template.html")).read_text(encoding="utf-8")
    for key, value in {
        "SPIKE": f"${n['spike']:,.0f}", "ENERGY": f"${n['energy']:.0f}", "SAVING": f"{n['saving']:.0%}",
        "SERVED": f"{n['served']:.0%}", "VIOLATIONS": str(n["violations"]), "WITH": str(n["with"]),
        "WITHOUT": str(n["without"]), "GAIN": f"+{n['with'] / n['without'] - 1:.0%}", "BUILDINGS": str(n["buildings"]),
        "SITES": str(n["sites"]), "MORE": f"{n['more']:.0%}", "MEDIAN": f"+{n['median']:.0%}",
        "BUILDING_PATH": c["building"], "EV_PATH": c["ev"], "LIMIT_Y": c["limit"], "PEAK_X": c["peak_x"],
        "PEAK_W": c["peak_w"], "REPO": REPO, "VIDEO": VIDEO, "EXTRA": f"${n['extra']:,.0f}",
        "SERVED36": f"{n['served36']:.0%}", "VIOL36": str(n["viol36"]),
    }.items():
        html = html.replace("{{" + key + "}}", value)
    assert "{{" not in html, "unfilled placeholder"
    out = ROOT / "docs" / "index.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in n.items()}))


if __name__ == "__main__":
    main()
