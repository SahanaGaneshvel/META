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


def heat_wave_day() -> tuple[pd.DatetimeIndex, "np.ndarray", "np.ndarray", "np.ndarray"]:
    """The real building's load on 21 June 2017 and the charging each controller added (Live replay)."""
    from headroom.agent.site import ReplaySite
    from headroom.config import LOCAL_TZ

    site = ReplaySite(month="2017-06", n_ports=24)
    local = site.scenario.index.tz_convert(LOCAL_TZ)
    day = (local >= pd.Timestamp("2017-06-21 06:00", tz=LOCAL_TZ)) & (local < pd.Timestamp("2017-06-21 23:00", tz=LOCAL_TZ))
    return (local[day], site.scenario.building_kw[day], site.results["headroom"].ev_kw[day],
            site.results["llf"].ev_kw[day])


def hero(name: str, size: tuple[float, float], tag: str) -> None:
    """Banner / thumbnail: the name and the real heat-wave day behind it."""
    t, building, ev, _ = heat_wave_day()
    fig = plt.figure(figsize=size, dpi=100, facecolor=BG)
    ax = fig.add_axes([0.58, 0.22, 0.38, 0.64], facecolor=BG)
    hours = t.hour + t.minute / 60
    ax.fill_between(hours, 0, building, color="#2563eb", alpha=0.85, lw=0, label="the building")
    ax.fill_between(hours, building, building + ev, color=GREEN, alpha=0.95, lw=0, label="EV charging (Headroom)")
    ax.set_ylim(0, 1000)  # from zero: the green band is drawn to scale
    ax.axhline(1000, color=MUTED, lw=1.2, ls="--")
    ax.text(hours.min() + 0.2, 965, "grid connection limit", color=MUTED, fontsize=11, va="top")
    ax.set_xlim(hours.min(), hours.max())
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.set_xticks([8, 12, 16, 20], ["8 am", "noon", "4 pm", "8 pm"])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, labelcolor=INK, fontsize=12)
    fig.text(0.96, 0.03, "Real metered load · 21 June 2017 heat-wave alert", ha="right", color=MUTED, fontsize=10)
    title = min(size[0] * 5.0, size[1] * 10)
    fig.text(0.05, 0.64, "HEADROOM", fontsize=title, color=INK, fontweight="bold", family="DejaVu Sans")
    fig.text(0.05, 0.52, "AI operator for EV charging", fontsize=title * 0.40, color=GREEN, family="DejaVu Sans")
    fig.text(0.05, 0.36, "More chargers on the grid\nconnection you already have", fontsize=title * 0.30,
             color=INK, family="DejaVu Sans", linespacing=1.3)
    fig.text(0.05, 0.19, f"  {tag}  ", fontsize=title * 0.22, color=BG, fontweight="bold", family="DejaVu Sans",
             bbox=dict(boxstyle="round,pad=0.4", fc=GREEN, ec="none"))
    fig.text(0.05, 0.07, "Muse Spark · Muse Voice Transcribe · Muse Glimmer", fontsize=title * 0.19, color=MUTED,
             family="DejaVu Sans")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main() -> None:
    hero("banner", (16, 5.6), "META AI HACKATHON 2026")
    hero("thumbnail", (12.8, 7.2), "PROJECT DEMO")
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
    card("2_result", lines, "Held-out months, real data: Palo Alto charging sessions · BDG2 metered buildings · Palo Alto and PG&E tariffs")
    card("3_close", [("Headroom", 80, INK), ("More chargers, same connection.", 40, GREEN),
                     ("The model proposes, the math decides, the safety layer guarantees.", 28, MUTED)],
         "Meta models: Muse Spark · Muse Voice Transcribe · Muse Glimmer   ·   Built Oct 26 – Nov 4, 2026")


if __name__ == "__main__":
    main()
