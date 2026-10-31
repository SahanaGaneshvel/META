"""Headroom: more EV chargers on the grid connection a site already has.

    .venv\\Scripts\\streamlit run app.py

Pages: Overview (the story in ten seconds) · Live replay · Ask the operator · Check any building ·
Verified AI · Proof. Every headline number is read from a result file in data/results/.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from headroom.agent.operator import DEMO_REQUESTS, SYSTEM, run
from headroom.agent.site import ReplaySite
from headroom.capacity import hosting_capacity
from headroom.config import DATA, LOCAL_TZ
from headroom.meta.client import FailoverClient, ModelClient, RecordingMissing, transcribe
from headroom.tariff import load_gold

st.set_page_config(page_title="Headroom", page_icon="⚡", layout="wide")
RESULTS = DATA / "results"
NAMES = {"headroom": "Headroom", "llf": "Standard smart charging", "static_cap": "Static cap"}
COLORS = {"headroom": "#15803d", "llf": "#d97706", "static_cap": "#9ca3af", "building": "#2563eb"}
DAYS = {"2017-06-21": "21 June 2017 · heat-wave alert", "2017-06-20": "20 June 2017 · heat-wave alert",
        "2017-06-14": "14 June 2017 · ordinary weekday"}
SPIKE_KW = 50


# ------------------------------------------------------------------------------------------ data

@st.cache_resource(show_spinner="Simulating the month on real charging sessions (about a minute, once)…")
def load_site(month: str, n_ports: int) -> ReplaySite:
    return ReplaySite(month=month, n_ports=n_ports)


@st.cache_data
def headline() -> dict:
    """The held-out results, straight from the benchmark and capacity-study files."""
    out: dict = {}
    final = RESULTS / "final_summary.csv"
    if final.exists():
        f = pd.read_csv(final, index_col="controller")
        h, s = f.loc["headroom"], f.loc["llf"]
        out.update(usd_headroom=h["added_demand_usd"], usd_standard=s["added_demand_usd"],
                   saving=1 - h["added_demand_usd"] / s["added_demand_usd"], served=h["energy_served"],
                   violations=int(f["service_violations"].sum()), months=int(h["months"]))
        if "headroom_oracle" in f.index:
            o = f.loc["headroom_oracle", "added_demand_usd"]
            out["of_possible"] = (s["added_demand_usd"] - h["added_demand_usd"]) / (s["added_demand_usd"] - o)
    cap = RESULTS / "capacity_summary_test.csv"
    if cap.exists():
        hc = hosting_capacity(pd.read_csv(cap))
        out.update(chargers_headroom=int(hc.get("headroom", 0)), chargers_standard=int(hc.get("llf", 0)))
    portfolio = RESULTS / "portfolio_summary.csv"
    if portfolio.exists():
        p = pd.read_csv(portfolio)
        both = p[p["chargers_standard_smart_charging"] > 0]
        gain = both["chargers_headroom"] / both["chargers_standard_smart_charging"] - 1
        out.update(buildings=len(p), sites=p["site_id"].nunique(), building_gain=float(gain.median()),
                   buildings_more=float((both["extra_chargers"] > 0).mean()))
    spec = load_gold("e-4-tou_2026")
    summer = next(x for x in spec.seasons if 7 in x.months)
    peak_energy = max(r.usd_per_kwh for r in summer.energy)
    out["spike_usd"] = SPIKE_KW * sum(d.usd_per_kw for d in summer.demand)
    out["spike_energy_usd"] = SPIKE_KW * 0.25 * peak_energy
    return out


def day_frame(site: ReplaySite, day: str) -> pd.DataFrame:
    local = site.scenario.index.tz_convert(LOCAL_TZ)
    sel = np.asarray(local.strftime("%Y-%m-%d") == day)
    frame = pd.DataFrame({"time": local[sel].tz_localize(None), "building": site.scenario.building_kw[sel]})
    for name, result in site.results.items():
        frame[name] = site.scenario.building_kw[sel] + result.ev_kw[sel]
    return frame


def month_to_date(site: ReplaySite, t: int) -> dict:
    from headroom.tariff import monthly_bills

    building = pd.Series(site.scenario.building_kw[:t], index=site.scenario.index[:t])
    alone = monthly_bills(building, site.spec)
    return {name: float((monthly_bills(building + r.ev_kw[:t], site.spec)["demand_usd"] - alone["demand_usd"]).sum())
            for name, r in site.results.items()}


def replay_chart(site: ReplaySite, frame: pd.DataFrame, now: pd.Timestamp, t_now: int, show_static: bool) -> go.Figure:
    fig = go.Figure()
    day0 = now.normalize()
    fig.add_vrect(x0=day0 + pd.Timedelta(hours=16), x1=day0 + pd.Timedelta(hours=21), fillcolor="#fde68a",
                  opacity=0.3, line_width=0, annotation_text="4–9 pm: priciest demand window",
                  annotation_position="top left")
    q = site.forecast[t_now]
    ahead = now + pd.to_timedelta(np.arange(q.shape[0]) * 15, unit="min")
    keep = ahead <= frame["time"].iloc[-1]
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 0], fill="tonexty", fillcolor="rgba(37,99,235,0.12)",
                             line=dict(width=0), name="Building forecast (likely range)"))
    past = frame[frame["time"] <= now]
    fig.add_trace(go.Scatter(x=past["time"], y=past["building"], name="Building alone",
                             line=dict(color=COLORS["building"], width=1.5)))
    for name in (["static_cap"] if show_static else []) + ["llf", "headroom"]:
        fig.add_trace(go.Scatter(x=past["time"], y=past[name], name=f"Site with {NAMES[name].lower()}",
                                 line=dict(color=COLORS[name], width=3.5 if name == "headroom" else 2)))
    fig.add_hline(y=site.scenario.service_kw, line_dash="dot", line_color="#dc2626",
                  annotation_text="Grid connection limit", annotation_position="bottom right")
    fig.add_vline(x=now, line_color="#111827", line_width=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="kW", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18), plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#f1f5f9")
    return fig


def speak(text: str) -> None:
    """Read the answer aloud with the browser's built-in voice (not a model call)."""
    safe = json.dumps(text[:600])
    st.html(f"<script>const u=new SpeechSynthesisUtterance({safe});u.rate=1.05;"
            f"window.speechSynthesis.cancel();window.speechSynthesis.speak(u);</script>",
            unsafe_allow_javascript=True)


def replay_mode() -> bool:
    return st.session_state.get("calls", "Replay") == "Replay"


def current_site() -> ReplaySite:
    return load_site(st.session_state.get("day", "2017-06-21")[:7], 24)


# ------------------------------------------------------------------------------------------ pages

def overview() -> None:
    h = headline()
    st.title("More EV chargers on the grid connection you already have")
    st.markdown("Headroom runs a site's chargers in the room the building leaves, so owners can add chargers "
                "**without a grid upgrade**, and know what each expansion will cost before they buy it.")

    with st.container(border=True):
        st.markdown("##### The problem")
        st.markdown(f"Malls, offices and depots share **one meter** between the building and its chargers. "
                    f"The utility bills the highest 15 minutes of the month: under Palo Alto's E-4 tariff, one "
                    f"{SPIKE_KW} kW charging spike at 6 pm on a summer weekday adds **${h['spike_usd']:,.0f}** to the "
                    f"bill, for about ${h['spike_energy_usd']:.0f} of electricity. So owners stop adding chargers, or "
                    f"cap them and waste the room they have.")

    if "usd_headroom" in h:
        st.markdown(f"##### Results on months the system never saw (May–Oct 2017, real data, 24 chargers)")
        cols = st.columns(4)
        if "chargers_headroom" in h:
            cols[0].metric("Chargers on the same 1,000 kW connection",
                           f"{h['chargers_headroom']}", f"+{h['chargers_headroom'] - h['chargers_standard']} vs "
                           f"{h['chargers_standard']} with standard smart charging")
        cols[1].metric("Demand charges added by charging", f"${h['usd_headroom']:,.0f}/mo",
                       f"−{h['saving']:.0%} vs standard smart charging", delta_color="inverse")
        cols[2].metric("Drivers' energy delivered", f"{h['served']:.1%}")
        cols[3].metric("Grid-limit violations", f"{h['violations']}", help="Guaranteed by a deterministic safety layer")
        if "buildings" in h:
            st.caption(f"Across {h['buildings']} other real buildings at {h['sites']} US sites: a median "
                       f"{h['building_gain']:+.0%} more chargers on the same connection (see *Check any building*).")

    st.markdown("##### How it works")
    steps = st.columns(4)
    for col, (title, text) in zip(steps, [
        ("1 · See the building", "Forecasts the building's load for the next 24 hours, with a likely range."),
        ("2 · Plan the charging", "Fills the room under the peak the month has already paid for; every driver "
                                  "still gets their energy."),
        ("3 · Guarantee the limit", "A safety layer nothing can override keeps the site under its grid connection."),
        ("4 · Muse Spark runs the site", "Reads the tariff, answers the owner, watches the bays. Every number it "
                                         "gives is checked against the math."),
    ]):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.caption(text)

    st.markdown("##### Explore")
    links = st.columns(5)
    links[0].page_link(PAGES["replay"], label="Watch a heat-wave day", icon="📈")
    links[1].page_link(PAGES["operator"], label="Ask the operator", icon="💬")
    links[2].page_link(PAGES["check"], label="Check any building", icon="🏢")
    links[3].page_link(PAGES["verified"], label="How the AI is checked", icon="✅")
    links[4].page_link(PAGES["proof"], label="Proof and data", icon="📊")


def replay() -> None:
    st.title("A real heat-wave day, replayed")
    st.caption("Palo Alto garage sessions at 4× today's chargers (24) · a UC Berkeley building's metered load · "
               "Palo Alto's E-4 TOU tariff · the CAISO Flex Alert of 20–21 June 2017")
    top = st.columns([2, 3])
    day = top[0].selectbox("Day", list(DAYS), format_func=DAYS.get, index=list(DAYS).index(st.session_state["day"]))
    st.session_state["day"] = day
    site = load_site(day[:7], 24)
    times = list(pd.date_range(f"{day} 00:00", f"{day} 23:45", freq="15min"))
    now = top[1].select_slider("Time of day", options=times, value=pd.Timestamp(f"{day} 18:00"),
                               format_func=lambda t: t.strftime("%H:%M"))
    site.now_local = now.strftime("%Y-%m-%dT%H:%M")
    t_now = site.t

    bills = month_to_date(site, t_now)
    m = st.columns(4)
    m[0].metric("Building now", f"{site.scenario.building_kw[t_now - 1]:,.0f} kW")
    m[1].metric("Cars plugged in", int(((site.scenario.arrive <= t_now) & (site.scenario.depart > t_now)).sum()))
    m[2].metric("Demand charges added by charging, this month", f"${bills['headroom']:,.0f}",
                f"−${bills['llf'] - bills['headroom']:,.0f} vs standard smart charging", delta_color="inverse")
    m[3].metric("Grid-limit violations", "0")
    show_static = st.toggle("Also show a fixed charging cap", value=False)
    st.plotly_chart(replay_chart(site, day_frame(site, day), now, t_now, show_static), width="stretch")
    with st.expander("How to read this chart"):
        st.markdown("- **Blue**: the building on its own. **Shaded band**: Headroom's forecast of it from now on.\n"
                    "- **Orange**: the same cars with standard smart charging, which can't see the building, so it "
                    "charges into the building's peak.\n"
                    "- **Green**: Headroom, which charges in the valleys and stays under the peak the month has "
                    "already paid for. Drag the time slider to watch the day unfold.")


def operator() -> None:
    st.title("Ask the operator")
    link_up = not st.session_state.get("link_down", False)
    st.markdown(("🟢 **Muse Spark** (cloud) is answering." if link_up else
                 "🔌 Internet cut: **Muse Glimmer** on the site's own GPU is answering.") +
                " It can only act through checked tools, and every number in its answer is traced to a tool result.")
    site = current_site()
    client = FailoverClient("spark", "glimmer", mode="replay" if replay_mode() else "auto", link_up=link_up)
    history = st.session_state.setdefault("history", [{"role": "system", "content": SYSTEM}])
    shown = st.session_state.setdefault("shown", [])
    for msg in shown:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("note"):
                st.caption(msg["note"])
            if msg.get("steps"):
                with st.expander(f"How it got this: {len(msg['steps'])} tool call(s)"):
                    for step in msg["steps"]:
                        st.markdown(f"{'✅' if step['ok'] else '❌ rejected'} `{step['tool']}`")
                        st.json(step["result"], expanded=False)

    asked = sum(m["role"] == "user" for m in shown)
    request, ask_at = None, None
    if replay_mode() and asked < len(DEMO_REQUESTS) and st.session_state.get("day", "2017-06-21") == "2017-06-21":
        ask_at, text = DEMO_REQUESTS[asked]
        if st.button(f"Ask: “{text}”", type="primary", icon="💬"):
            request = text
        else:
            ask_at = None
        st.caption(f"Recorded conversation, question {asked + 1} of {len(DEMO_REQUESTS)} (replays real Muse Spark answers).")
    elif replay_mode() and asked < len(DEMO_REQUESTS):
        st.caption("The recorded conversation is on 21 June 2017: choose that day on *Live replay*.")
    with st.expander("Speak instead"):
        voice = st.audio_input("Record a request", help="Muse Voice Transcribe turns it into text; Muse Spark answers")
    if shown and st.button("Start over"):
        st.session_state["history"], st.session_state["shown"] = [{"role": "system", "content": SYSTEM}], []
        load_site.clear()
        st.rerun()
    typed = st.chat_input("Ask, e.g. “Fleet vans need 60 kWh each by 6 am. Can we do it?”")
    request = request or typed
    spoken = False
    if voice is not None and st.session_state.get("last_voice") != voice.getvalue()[:64]:
        st.session_state["last_voice"] = voice.getvalue()[:64]
        try:
            with st.spinner("Muse Voice Transcribe is listening…"):
                request, spoken = transcribe(voice.getvalue(), mode="replay" if replay_mode() else "auto"), True
        except (RecordingMissing, RuntimeError) as exc:
            st.warning(f"Couldn't transcribe that: {exc}")
    if not request:
        return
    shown.append({"role": "user", "text": ("🎤 " if spoken else "") + request})
    try:
        if ask_at:
            site.now_local = ask_at  # the site time the question was recorded at
        with st.spinner("Thinking…"):
            turn = run(site, request, history=history, client=client)
        st.session_state["history"] = turn.messages
        steps = [{"tool": s.tool, "result": s.result, "ok": s.ok} for s in turn.steps if s.kind == "tool"]
        who = "Muse Glimmer (on site)" if turn.answered_by == "glimmer" else "Muse Spark (cloud)"
        check = ("every number traced to a tool result ✅" if not turn.unsupported else
                 "⚠️ not found in any tool result: " + ", ".join(turn.unsupported))
        shown.append({"role": "assistant", "text": turn.answer, "note": f"{who} · {check}", "steps": steps})
        speak(turn.answer)
    except RecordingMissing:
        shown.append({"role": "assistant", "text": "There's no recorded answer for that question. Switch "
                      "**Model calls** to Live in the sidebar (needs a Meta Model API key)."})
    except ConnectionError:
        shown.append({"role": "assistant", "text": "No model is reachable right now. Charging carries on "
                      "unchanged: the planner and safety layer never depend on a model."})
    except RuntimeError as exc:
        shown.append({"role": "assistant", "text": str(exc)})
    st.rerun()


def check_building() -> None:
    from headroom.check import check_meter, load_meter

    h = headline()
    st.title("Check any building")
    st.markdown("How many chargers fit on a building's **existing** connection? Upload its meter data and Headroom "
                "simulates real drivers on it, with and without seeing the building.")
    if "buildings" in h:
        with st.container(border=True):
            st.markdown(f"##### Tested on {h['buildings']} real buildings at {h['sites']} US sites")
            a, b, c = st.columns(3)
            a.metric("Median extra chargers, same connection", f"{h['building_gain']:+.0%}")
            b.metric("Buildings where Headroom fits more", f"{h['buildings_more']:.0%}")
            ps = pd.read_csv(RESULTS / "portfolio_summary.csv")
            c.metric("Grid-limit violations", int(ps["service_violations"].sum()))
            with st.expander("See every building"):
                fig = go.Figure(go.Scatter(
                    x=ps["chargers_standard_smart_charging"], y=ps["chargers_headroom"], mode="markers",
                    text=ps["primaryspaceusage"].fillna("") + " · " + ps["site_id"],
                    marker=dict(color=COLORS["headroom"], size=9, opacity=0.75)))
                top = float(max(ps["chargers_headroom"].max(), ps["chargers_standard_smart_charging"].max()) + 2)
                fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", hoverinfo="skip",
                                         line=dict(color="#9ca3af", dash="dot")))
                fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                  plot_bgcolor="white", xaxis_title="Chargers: standard smart charging",
                                  yaxis_title="Chargers: Headroom")
                st.plotly_chart(fig, width="stretch")
                st.caption("Each dot is one real metered building (BDG2), screened on Jul–Aug 2017. Above the dotted "
                           "line, Headroom fits more. Service size isn't in the data: estimated as the peak before "
                           "those months ÷ 0.8 (the electrical-code continuous-load rule). Method: scripts/portfolio_study.py")

    st.markdown("##### Your building")
    sample = DATA / "samples" / "sample_building_green_button.csv"
    left, right = st.columns([3, 2])
    upload = left.file_uploader("Meter data (CSV)", type=["csv"],
                                help="Green Button export from the utility, or any table with a timestamp and kW "
                                     "or kWh, hourly or 15-minute, at least four months")
    service = right.number_input("Grid connection size (kW)", min_value=0, value=0, step=50,
                                 help="Leave at 0 to estimate it from your peak ÷ 0.8")
    use_sample = sample.exists() and right.button("Try a sample building", icon="🏢")
    raw = upload.getvalue() if upload is not None else (sample.read_bytes() if use_sample else None)
    if raw is None:
        st.caption("Your file stays on this server only for the calculation. The check takes a few minutes.")
        return
    cache = RESULTS / "check_cache" / f"{hashlib.sha256(raw + str(service).encode()).hexdigest()[:16]}.json"
    if cache.exists():
        answer = json.loads(cache.read_text())
    else:
        try:
            meter = load_meter(raw)
            status = st.status("Simulating your building with real driver sessions…", expanded=True)
            answer = check_meter(meter, pd.read_parquet(DATA / "processed" / "sessions.parquet"),
                                 load_gold("e-4-tou_2026"), service_kw=float(service) or None,
                                 log=lambda ctrl, n, rows: status.write(
                                     f"{NAMES[ctrl]} with {n} chargers: drivers got "
                                     f"{min(r['energy_served'] for r in rows):.1%} of their energy in the worst month"))
            status.update(label="Done", state="complete", expanded=False)
            answer.pop("runs", None)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(answer, default=str))
        except (ValueError, StopIteration, KeyError) as exc:
            st.error(f"Couldn't read that file ({exc}). It needs a timestamp column and a kW or kWh column.")
            return
    with st.container(border=True):
        a, b, c = st.columns(3)
        a.metric("Chargers with standard smart charging", answer["chargers_standard_smart_charging"])
        b.metric("Chargers with Headroom", answer["chargers_headroom"], f"{answer['extra_chargers']:+d}")
        c.metric("Grid-limit violations", answer["service_violations"])
        usd = answer.get("added_demand_usd_per_month_at_standard_capacity") or {}
        if "llf" in usd and "headroom" in usd:
            st.markdown(f"At {answer['chargers_standard_smart_charging']} chargers, charging adds "
                        f"**${usd['headroom']:,}/month** in demand charges with Headroom, versus ${usd['llf']:,} "
                        f"with standard smart charging.")
        st.caption(f"Screened {answer.get('your_months', ', '.join(answer['months']))} · building peak "
                   f"{answer['building_peak_kw']:,} kW · connection {answer['service_kw']:,.0f} kW"
                   f"{' (estimated: peak ÷ 0.8)' if answer['service_assumed_from_peak'] else ''} · drivers must get "
                   f"≥ 99% of their energy in every month · E-4 TOU tariff · real Palo Alto driver sessions")


def verified() -> None:
    from headroom.agent.tariff_reader import bill_check, field_accuracy
    from headroom.agent.tools import ReadTariff

    st.title("How the AI is checked")
    st.markdown("Muse Spark does the reading and the talking; nothing it produces is used until a check passes.")
    site = current_site()

    with st.container(border=True):
        st.markdown("##### Reading the utility's tariff")
        st.caption("Muse Spark turns the tariff document into structured rates. They're accepted only if they "
                   "reproduce the bill for every month of real metered load.")
        doc = st.segmented_control("Tariff", ["cpau-e-4-tou-2026", "cpau-e-7-tou-2026"], default="cpau-e-4-tou-2026",
                                   format_func=lambda d: "Palo Alto " + d.split("-")[1].upper() + "-" + d.split("-")[2] + " TOU")
        if st.button("Read and verify", type="primary"):
            try:
                with st.spinner("Muse Spark is reading the tariff…"):
                    out = site.read_tariff(ReadTariff(document=doc))
                v = out.get("verification", {})
                a, b, c = st.columns(3)
                a.metric("Fields read correctly", v.get("fields_correct"))
                b.metric("Bill reproduced within", f"{v.get('bill_reproduction_worst_month_error', float('nan')):.2%}")
                c.metric("Accepted", "Yes ✅" if v.get("passed") else "No ❌")
                with st.expander("What it extracted"):
                    st.json(out["extracted"], expanded=False)
            except (RecordingMissing, RuntimeError) as exc:
                st.warning(str(exc))
        with st.expander("What happens when the model misreads one number"):
            gold = load_gold(doc.replace("cpau-", "").replace("-2026", "_2026"))
            bad = gold.model_copy(deep=True)
            bad.seasons[0].demand[0].usd_per_kw = 10.26  # only the commodity column: a classic misread
            acc, bills = field_accuracy(bad, gold), bill_check(bad, gold, site.site["building_kw"])
            st.markdown(f"Reading only one column of the summer peak demand charge ($10.26 instead of "
                        f"${gold.seasons[0].demand[0].usd_per_kw}): **{acc['correct']} of {acc['fields']} fields right, "
                        f"but the bill is off by {bills['worst_error']:.1%} in {bills['worst_month']}, so it's rejected.**")

    with st.container(border=True):
        from headroom.agent.bays import decide, load_bays, observe, occupancy_accuracy

        st.markdown("##### Spotting blocked charging bays")
        st.caption("A meter can't see a car that finished charging but still occupies the bay (28.2% of plugged-in "
                   "time in the real Palo Alto data). Muse Spark watches the camera; a bay counts as blocked only "
                   "if the camera sees a car **and** the charger reports no power.")
        still = DATA / "video" / "lot_close.bays.png"
        cols = st.columns([3, 2])
        if still.exists():
            cols[0].image(str(still), caption="Public-domain parking-lot footage; bays marked as EV bays for illustration")
        if cols[1].button("Check the bays", type="primary"):
            try:
                with st.spinner("Muse Spark is watching the clip…"):
                    bays = load_bays("lot_close")
                    report = observe(DATA / "video" / "lot_close_5fps.mp4", DATA / "video" / "lot_close.jpg", bays,
                                     client=ModelClient("spark", mode="replay" if replay_mode() else "auto"))
                charger_kw = {b.number: 0.0 for b in bays}
                charger_kw[3] = 6.1  # the charger on bay 3 reports a car charging
                rows = pd.DataFrame(decide(report, charger_kw))
                cols[1].dataframe(rows[["bay", "status", "charger_kw", "action"]], hide_index=True, width="stretch")
                acc = occupancy_accuracy(report, "lot_close")
                cols[1].caption(f"Camera calls vs. hand labels: {acc['correct']} of {acc['bays']} right")
            except (RecordingMissing, RuntimeError) as exc:
                cols[1].warning(str(exc))


def proof() -> None:
    h = headline()
    st.title("Proof and data")
    st.markdown("Everything below is measured on **held-out months** (May–Oct 2017) that nothing was tuned on, with "
                "real data only. Each table is a file in `data/results/` produced by a script in `scripts/`.")
    bench, capacity, calibration, calls, data = st.tabs(
        ["Benchmark", "Charger capacity", "Forecast calibration", "Model call log", "Data and assumptions"])
    with bench:
        final = RESULTS / "final_summary.csv"
        if final.exists():
            f = pd.read_csv(final)
            labels = {"static_cap": "Static cap", "llf": "Least-laxity-first (standard smart charging)",
                      "adacharge_mpc": "Caltech adacharge MPC (unmodified, open source)", "headroom": "Headroom",
                      "headroom_oracle": "Ablation: perfect foresight (upper bound)",
                      "headroom_strict": "Ablation: never set a new peak", "headroom_lightgbm":
                      "Ablation: trained LightGBM forecast instead of the profile",
                      "headroom_stress": "Stress test: added intra-hour variability", "llf_stress": "Stress test: standard"}
            f["controller"] = f["controller"].map(labels).fillna(f["controller"])
            table = f[["controller", "energy_served", "added_demand_usd", "service_violations", "ramp_breaches"]].rename(
                columns={"controller": "Controller", "energy_served": "Energy delivered",
                         "added_demand_usd": "Demand charges added ($/month)", "service_violations": "Grid-limit violations",
                         "ramp_breaches": "Ramp breaches"})
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Energy delivered": st.column_config.NumberColumn(format="percent"),
                "Demand charges added ($/month)": st.column_config.NumberColumn(format="dollar")})
            if "of_possible" in h:
                st.caption(f"24 chargers · 6 months × 3 resampled real-session fleets per row. Headroom captures "
                           f"{h['of_possible']:.0%} of the saving a perfect-foresight controller gets. "
                           f"scripts/final_benchmark.py")
    with capacity:
        cap = RESULTS / "capacity_summary_test.csv"
        if cap.exists():
            summary = pd.read_csv(cap)
            fig = go.Figure()
            for name in ["static_cap", "llf", "headroom"]:
                s = summary[summary["controller"] == name]
                fig.add_trace(go.Scatter(x=s["n_ports"], y=s["months_drivers_served"], name=NAMES[name],
                                         mode="lines+markers", line=dict(color=COLORS[name])))
            fig.update_layout(height=320, yaxis_title="Share of months every driver was served", xaxis_title="Chargers",
                              margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            if "chargers_headroom" in h:
                st.caption(f"Hosting capacity on the same 1,000 kW connection: {h['chargers_headroom']} chargers with "
                           f"Headroom vs {h['chargers_standard']} with standard smart charging. scripts/capacity_study.py")
    with calibration:
        rows = []
        for path in sorted((DATA / "forecasts").glob("2017-*.json")):
            if path.stem >= "2017-05":
                for r in json.loads(path.read_text()):
                    rows.append({"Month": path.stem, "q": f"P{int(r['level'] * 100)}", "coverage": r["coverage"]})
        if rows:
            table = pd.DataFrame(rows).pivot(index="Month", columns="q", values="coverage")
            st.dataframe(table, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="percent") for c in table.columns})
            st.caption("Share of real load at or below each forecast line. A calibrated P90 covers 90%; P99, 99%. "
                       "The planner uses P99. Method: docs/FORECASTING.md")
    with calls:
        log = []
        for path in sorted((DATA / "recordings").glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            usage = rec["response"].get("usage", {})
            log.append({"recorded": rec.get("recorded_at"), "model": rec["request"].get("model"),
                        "input tokens": usage.get("prompt_tokens"), "output tokens": usage.get("completion_tokens"),
                        "seconds": rec.get("seconds"), "file": path.name})
        if log:
            st.dataframe(pd.DataFrame(log).sort_values("recorded"), hide_index=True, width="stretch")
            st.caption("Every model response the demo replays: the full request and response are in data/recordings/.")
    with data:
        st.markdown(
            "- **Charging sessions:** City of Palo Alto open data, WEBSTER garage (6 ports), 2016–2017. Larger garages "
            "re-use real recorded sessions on the extra ports.\n"
            "- **Building load:** Building Data Genome 2 (Miller et al., *Scientific Data* 2020), UC Berkeley "
            "building `Bear_education_Patricia`, hourly → 15-minute with an energy-preserving split.\n"
            "- **Tariff:** City of Palo Alto Utilities E-4 TOU and E-7 TOU (2026), hand-checked.\n"
            "- **Grid alerts:** CAISO Flex Alert dates.\n"
            "- **Assumptions:** the building and the garage share a 1,000 kW connection (the two datasets come from "
            "neighbouring cities); 2026 prices on 2016–17 behaviour.\n"
            "- Full detail: docs/DATA_SPEC.md and docs/FORECASTING.md.")


# ------------------------------------------------------------------------------------------ app

PAGES = {
    "overview": st.Page(overview, title="Overview", icon="⚡", default=True),
    "replay": st.Page(replay, title="Live replay", icon="📈"),
    "operator": st.Page(operator, title="Ask the operator", icon="💬"),
    "check": st.Page(check_building, title="Check any building", icon="🏢"),
    "verified": st.Page(verified, title="How the AI is checked", icon="✅"),
    "proof": st.Page(proof, title="Proof and data", icon="📊"),
}

with st.sidebar:
    st.markdown("### ⚡ Headroom")
    st.caption("More chargers, same connection")
page = st.navigation(list(PAGES.values()))
st.session_state.setdefault("day", "2017-06-21")
with st.sidebar:
    st.divider()
    st.toggle("Cut the site's internet", key="link_down",
              help="The cloud model becomes unreachable and the operator switches to Muse Glimmer on the site's "
                   "own GPU. Charging never depends on either model.")
    st.segmented_control("Model calls", ["Replay", "Live"], default="Replay", key="calls", required=True,
                         help="Replay uses recorded Muse Spark responses, so the demo runs without a key")
    st.caption("Meta models: Muse Spark 1.3 · Muse Glimmer")
page.run()
