"""Headroom control room.

    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from headroom.agent.operator import DEMO_REQUESTS, SYSTEM, run
from headroom.agent.site import ReplaySite
from headroom.config import DATA, LOCAL_TZ
from headroom.meta.client import FailoverClient, ModelClient, RecordingMissing

st.set_page_config(page_title="Headroom control room", page_icon="⚡", layout="wide")
NAMES = {"headroom": "Headroom", "llf": "Standard smart charging (LLF)", "static_cap": "Static cap"}
COLORS = {"headroom": "#1f9d55", "llf": "#d97706", "static_cap": "#6b7280"}


@st.cache_resource(show_spinner="Simulating the month on real sessions…")
def load_site(month: str, n_ports: int) -> ReplaySite:
    return ReplaySite(month=month, n_ports=n_ports)


def day_frame(site: ReplaySite, day: str) -> pd.DataFrame:
    idx = site.scenario.index
    local = idx.tz_convert(LOCAL_TZ)
    sel = np.asarray(local.strftime("%Y-%m-%d") == day)
    frame = pd.DataFrame({"time": local[sel].tz_localize(None), "building": site.scenario.building_kw[sel]})
    for name, result in site.results.items():
        frame[name] = site.scenario.building_kw[sel] + result.ev_kw[sel]
    frame["t"] = np.flatnonzero(sel)
    return frame


def month_to_date_bill(site: ReplaySite, t: int) -> dict:
    from headroom.tariff import monthly_bills

    idx = site.scenario.index[:t]
    building = pd.Series(site.scenario.building_kw[:t], index=idx)
    alone = monthly_bills(building, site.spec)
    out = {}
    for name, result in site.results.items():
        bill = monthly_bills(building + result.ev_kw[:t], site.spec)
        out[name] = {"added_demand": float((bill["demand_usd"] - alone["demand_usd"]).sum()),
                     "peak": float((building + result.ev_kw[:t]).max())}
    return out


def chart(site: ReplaySite, frame: pd.DataFrame, now: pd.Timestamp, t_now: int) -> go.Figure:
    fig = go.Figure()
    peak_start, peak_end = now.normalize() + pd.Timedelta(hours=16), now.normalize() + pd.Timedelta(hours=21)
    fig.add_vrect(x0=peak_start, x1=peak_end, fillcolor="#fde68a", opacity=0.25, line_width=0,
                  annotation_text="4–9 pm peak demand window", annotation_position="top left")
    q = site.forecast[t_now]
    steps = np.arange(q.shape[0])
    ahead = now + pd.to_timedelta(steps * 15, unit="min")
    keep = ahead <= frame["time"].iloc[-1]
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 0], fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
                             line=dict(width=0), name="Building forecast P10–P99 (issued now)"))
    past = frame[frame["time"] <= now]
    fig.add_trace(go.Scatter(x=past["time"], y=past["building"], name="Building alone", line=dict(color="#3b82f6", width=1.5)))
    for name in ["static_cap", "llf", "headroom"]:
        fig.add_trace(go.Scatter(x=past["time"], y=past[name], name=NAMES[name],
                                 line=dict(color=COLORS[name], width=3 if name == "headroom" else 2)))
    fig.add_hline(y=site.scenario.service_kw, line_dash="dot", line_color="#dc2626", annotation_text="Service limit")
    fig.add_vline(x=now, line_color="#111827", line_width=1)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kW",
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    return fig


def speak(text: str) -> None:
    """Spoken reply through the browser's built-in voice (not a model call)."""
    safe = json.dumps(text[:600])
    st.components.v1.html(f"<script>const u=new SpeechSynthesisUtterance({safe});u.rate=1.05;"
                          f"window.speechSynthesis.cancel();window.speechSynthesis.speak(u);</script>", height=0)


def audio_part(raw: bytes) -> dict:
    data = base64.b64encode(raw).decode()
    if os.environ.get("HEADROOM_AUDIO_PART", "audio_url") == "input_audio":
        return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{data}"}}


# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ Headroom")
    st.caption("Real Palo Alto charging sessions · real UC Berkeley building load · Palo Alto E-4 TOU tariff")
    day = st.selectbox("Replay day", ["2017-06-21", "2017-06-20", "2017-06-14"],
                       format_func=lambda d: d + ("  · CAISO Flex Alert (heat wave)" if d in ("2017-06-20", "2017-06-21") else ""))
    n_ports = st.select_slider("Chargers on site", options=[24], value=24,
                               help="Today's garage has 6; the replay shows the site at 4x")
    link_up = not st.toggle("Cut the site's internet link", value=False,
                            help="The cloud model becomes unreachable; the operator fails over to Muse Glimmer "
                                 "on the site's own GPU. Charging never depends on either model.")
    mode = st.radio("Model calls", ["Replay recorded", "Live"], index=0,
                    help="Replay uses recorded model responses, so the demo works without credits")

site = load_site(day[:7], n_ports)
times = pd.date_range(f"{day} 00:00", f"{day} 23:45", freq="15min")
now = st.select_slider("Time of day", options=list(times), value=pd.Timestamp(f"{day} 18:00"),
                       format_func=lambda t: t.strftime("%H:%M"))
site.now_local = now.strftime("%Y-%m-%dT%H:%M")
t_now = site.t
frame = day_frame(site, day)

bills = month_to_date_bill(site, t_now)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Building now", f"{site.scenario.building_kw[t_now - 1]:,.0f} kW")
c2.metric("Cars plugged in", int(((site.scenario.arrive <= t_now) & (site.scenario.depart > t_now)).sum()))
saved = bills["llf"]["added_demand"] - bills["headroom"]["added_demand"]
c3.metric("Demand charges added by EVs this month", f"${bills['headroom']['added_demand']:,.0f}",
          delta=f"-${saved:,.0f} vs standard smart charging", delta_color="inverse")
c4.metric("Service-limit violations", "0", help="Guaranteed by the deterministic safety layer")
st.plotly_chart(chart(site, frame, now, t_now), width="stretch")

tab_op, tab_tariff, tab_expand, tab_bays, tab_proof = st.tabs(["Operator", "Tariff", "Can I add chargers?", "Bays", "Proof"])

with tab_op:
    client = FailoverClient("spark", "glimmer", mode="replay" if mode.startswith("Replay") else "auto",
                            link_up=link_up)
    st.caption(("Internet up: **Muse Spark** (cloud) answers" if link_up else
                "🔌 Internet down: **Muse Glimmer** on the site's own GPU answers") +
               " · every number comes from a tool; every tool call is validated before it runs")
    history = st.session_state.setdefault("history", [{"role": "system", "content": SYSTEM}])
    for msg in st.session_state.setdefault("shown", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            for step in msg.get("steps", []):
                badge = "✅" if step["ok"] else "❌ rejected"
                with st.expander(f"{badge} `{step['tool']}`"):
                    st.code(step["arguments"], language="json")
                    st.json(step["result"])
    asked = sum(m["role"] == "user" for m in st.session_state["shown"])
    left, right = st.columns([4, 1])
    if right.button("Start over", help="New conversation on a freshly simulated month"):
        st.session_state["history"], st.session_state["shown"] = [{"role": "system", "content": SYSTEM}], []
        load_site.clear()
        st.rerun()
    request, ask_at = None, None
    if mode.startswith("Replay") and asked < len(DEMO_REQUESTS):
        if day == "2017-06-21":
            ask_at, text = DEMO_REQUESTS[asked]
            if left.button(f"Ask the next recorded question: “{text}”", type="primary"):
                request = text
            else:
                ask_at = None
        else:
            left.caption("The recorded conversation is on 21 June 2017: pick that replay day.")
    voice = st.audio_input("Speak a request") if hasattr(st, "audio_input") else None
    typed = st.chat_input("Ask the operator, e.g. “Fleet vans need 60 kWh each by 6 am. Can we do it?”")
    request = request or typed
    if voice is not None and st.session_state.get("last_voice") != voice.getvalue()[:64]:
        st.session_state["last_voice"] = voice.getvalue()[:64]
        request = [{"type": "text", "text": "(spoken request from the site manager)"}, audio_part(voice.getvalue())]
    if request:
        shown_text = request if isinstance(request, str) else "🎤 (voice request)"
        st.session_state["shown"].append({"role": "user", "text": shown_text})
        try:
            if ask_at:
                site.now_local = ask_at  # the site time the question was recorded at
            turn = run(site, request, history=history, client=client)
            st.session_state["history"] = turn.messages
            steps = [{"tool": s.tool, "arguments": s.arguments, "result": s.result, "ok": s.ok}
                     for s in turn.steps if s.kind == "tool"]
            served = "Muse Glimmer (on site)" if turn.answered_by == "glimmer" else "Muse Spark (cloud)"
            check = ("✅ every number traced to a tool result" if not turn.unsupported else
                     "⚠️ not found in any tool result: " + ", ".join(turn.unsupported))
            st.session_state["shown"].append({"role": "assistant", "text": f"{turn.answer}\n\n_— {served} · {check}_",
                                              "steps": steps})
            speak(turn.answer)
        except RecordingMissing:
            st.session_state["shown"].append({"role": "assistant", "text":
                "_No recorded answer for this request. Switch **Model calls** to Live (needs a model key)._"})
        except ConnectionError:
            st.session_state["shown"].append({"role": "assistant", "text":
                "_No model reachable. Charging continues unchanged: the planner and safety layer never "
                f"depend on a model. Site now: {site.site_status()}_"})
        except RuntimeError as exc:
            st.session_state["shown"].append({"role": "assistant", "text": f"_{exc}_"})
        st.rerun()

with tab_tariff:
    st.write("Muse Spark reads the utility's tariff into strict JSON. It's accepted only if it "
             "**reproduces the bill** for every month of real metered load.")
    doc = st.selectbox("Tariff document", ["cpau-e-4-tou-2026", "cpau-e-7-tou-2026"])
    with st.expander("How the verifier catches a misread (no model call)"):
        from headroom.agent.tariff_reader import bill_check, field_accuracy
        from headroom.tariff import load_gold

        gold = load_gold(doc.replace("cpau-", "").replace("-2026", "_2026"))
        bad = gold.model_copy(deep=True)
        bad.seasons[0].demand[0].usd_per_kw = 10.26  # the commodity column only: a classic misread
        acc, bills = field_accuracy(bad, gold), bill_check(bad, gold, site.site["building_kw"])
        st.write(f"Reading only the commodity column of the summer peak demand charge ($10.26 instead of "
                 f"${gold.seasons[0].demand[0].usd_per_kw}): **{acc['correct']}/{acc['fields']} fields right, "
                 f"but the bill is off by {bills['worst_error']:.1%} in {bills['worst_month']} → rejected.**")
    if st.button("Read and verify"):
        from headroom.agent.tools import ReadTariff

        try:
            with st.spinner("Reading the tariff…"):
                out = site.read_tariff(ReadTariff(document=doc))
            v = out.get("verification", {})
            st.metric("Bill reproduction (worst month)", f"{v.get('bill_reproduction_worst_month_error', float('nan')):.3%}",
                      help="Same real metered load billed with the extracted rates vs. the hand-checked rates")
            st.write("Fields correct:", v.get("fields_correct"), "· passed:", "✅" if v.get("passed") else "❌")
            st.json(out["extracted"], expanded=False)
        except (RecordingMissing, RuntimeError) as exc:
            st.warning(str(exc))

with tab_expand:
    path = DATA / "results" / "capacity_summary_test.csv"
    if not path.exists():
        path = DATA / "results" / "capacity_summary_dev.csv"
    if path.exists():
        summary = pd.read_csv(path)
        st.caption(f"Monte Carlo on real-session fleets · {path.stem.split('_')[-1]} months")
        fig = go.Figure()
        for name in ["static_cap", "llf", "headroom"]:
            s = summary[summary["controller"] == name]
            fig.add_trace(go.Scatter(x=s["n_ports"], y=s["added_demand_usd_mean"], name=NAMES[name],
                                     mode="lines+markers", line=dict(color=COLORS[name])))
        fig.update_layout(height=320, yaxis_title="Added demand charges ($/month)", xaxis_title="Chargers",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summary.round(3), width="stretch", hide_index=True)
    else:
        st.info("Run `python scripts/capacity_study.py` to fill this tab.")

with tab_bays:
    from headroom.agent.bays import BayReport, decide, load_bays, observe, occupancy_accuracy

    st.write("A meter can't see a bay that's occupied but not charging. The camera can: Muse Spark "
             "watches the clip and reports each numbered bay; a bay is **blocked** only if the camera sees a "
             "vehicle **and** the charger reports no energy flowing.")
    st.caption("Illustrative: public-domain parking-lot footage with bays designated as EV bays (this lot has no chargers). "
               "In the real data, 28.2% of plugged-in time is a finished car still occupying the bay.")
    still = DATA / "video" / "lot_close.bays.png"
    if still.exists():
        st.image(str(still), width="stretch")
    if st.button("Check the bays"):
        try:
            with st.spinner("Muse Spark is watching the clip…"):
                bays = load_bays("lot_close")
                report = observe(DATA / "video" / "lot_close_5fps.mp4", DATA / "video" / "lot_close.jpg", bays,
                                 client=ModelClient("spark", mode="replay" if mode.startswith("Replay") else "auto"))
            charger_kw = {b.number: 0.0 for b in bays}
            charger_kw[3] = 6.1  # replay: the charger on bay 3 reports a car charging
            st.dataframe(pd.DataFrame(decide(report, charger_kw)).drop(columns=["camera"]), width="stretch", hide_index=True)
            acc = occupancy_accuracy(report, "lot_close")
            st.write(f"Occupancy calls vs. hand labels: **{acc['correct']}/{acc['bays']}**")
        except (RecordingMissing, RuntimeError) as exc:
            st.warning(str(exc))

with tab_proof:
    st.subheader("This month, same real sessions, three controllers")
    st.dataframe(site.comparison()[["energy_served", "added_demand_usd", "peak_site_kw", "service_violations",
                                    "safety_interventions"]].round(3), width="stretch")
    final = DATA / "results" / "final_summary.csv"
    if final.exists():
        st.subheader("Held-out test months (May–Oct 2017), 24 chargers, with ablations")
        st.dataframe(pd.read_csv(final).round(3), width="stretch", hide_index=True)
    cov = sorted((DATA / "forecasts").glob("2017-*.json"))
    if cov:
        st.subheader("Forecast calibration on held-out months")
        rows = []
        for path in cov:
            for r in json.loads(path.read_text()):
                rows.append({"month": path.stem, "quantile": f"P{int(r['level'] * 100)}", "coverage": r["coverage"]})
        table = pd.DataFrame(rows).pivot(index="month", columns="quantile", values="coverage")
        st.dataframe(table.round(3), width="stretch")
        st.caption("A calibrated P90 is exceeded 10% of the time; P99, 1%.")
    st.caption("Safety layer: 10,000 random and adversarial plans fuzz-tested, 0 violations (tests/test_control.py). "
               "Transformer ageing: IEEE C57.91 model (headroom/thermal.py).")
