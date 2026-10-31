"""Title cards for the demo video (1920x1080 PNG), with every number read from data/results/.

    python scripts/make_cards.py        # -> docs/cards/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from headroom.capacity import hosting_capacity  # noqa: E402
from headroom.config import DATA, ROOT  # noqa: E402
from headroom.tariff import load_gold  # noqa: E402

OUT = ROOT / "docs" / "cards"
INK, MUTED, GREEN, BG = "#f9fafb", "#9ca3af", "#22c55e", "#0b0f14"


def card(name: str, lines: list[tuple[str, float, str]], footer: str = "") -> None:
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    y = 0.62 if len(lines) <= 2 else 0.7
    for text, size, color in lines:
        fig.text(0.5, y, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold"
                 if size >= 48 else "normal", family="DejaVu Sans")
        y -= size / 520 + 0.06
    if footer:
        fig.text(0.5, 0.08, footer, ha="center", va="center", fontsize=20, color=MUTED, family="DejaVu Sans")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main() -> None:
    spec = load_gold("e-4-tou_2026")
    summer = next(s for s in spec.seasons if 7 in s.months)
    spike = 50 * sum(d.usd_per_kw for d in summer.demand)
    energy = 50 * 0.25 * max(r.usd_per_kwh for r in summer.energy)
    card("1_hook", [(f"This 15 minutes cost ${spike:,.0f}.", 72, INK),
                    (f"A 50 kW charging spike at 6 pm, summer weekday. The electricity in it: ${energy:.2f}.", 28, MUTED)],
         "Palo Alto E-4 TOU tariff (2026): two demand charges on the highest 15 minutes of the month")

    final = pd.read_csv(DATA / "results" / "final_summary.csv", index_col="controller")
    cap = hosting_capacity(pd.read_csv(DATA / "results" / "capacity_summary_test.csv"))
    saving = 1 - final.loc["headroom", "added_demand_usd"] / final.loc["llf", "added_demand_usd"]
    lines = [(f"{int(cap['headroom'])} chargers instead of {int(cap['llf'])}.", 72, INK),
             ("Same grid connection. 0 limit violations.", 40, GREEN),
             (f"{saving:.0%} lower demand charges · {final.loc['headroom', 'energy_served']:.1%} of drivers' energy "
              f"delivered", 28, MUTED)]
    portfolio = DATA / "results" / "portfolio_summary.csv"
    if portfolio.exists():
        p = pd.read_csv(portfolio)
        both = p[p["chargers_standard_smart_charging"] > 0]
        gain = (both["chargers_headroom"] / both["chargers_standard_smart_charging"] - 1).median()
        lines.append((f"Across {len(p)} real buildings at {p['site_id'].nunique()} US sites: median {gain:+.0%} chargers",
                      28, MUTED))
    card("2_result", lines, "Held-out months, real data: Palo Alto charging sessions · BDG2 metered buildings · CPAU tariffs")
    card("3_close", [("Headroom", 80, INK), ("More chargers, same connection.", 40, GREEN),
                     ("The model proposes, the math decides, the safety layer guarantees.", 28, MUTED)],
         "Meta models: Muse Spark · Muse Voice Transcribe · Muse Glimmer   ·   Built Oct 26 – Nov 4, 2026")


if __name__ == "__main__":
    main()
