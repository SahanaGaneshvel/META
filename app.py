"""Headroom: more EV chargers on the grid connection a site already has.

    .venv\\Scripts\\streamlit run app.py

Start · Watch it run · Ask the operator · What the camera sees · Check your building · How the AI is
checked · Proof. Every headline number is read from a result file in data/results/; every model answer
is a real recorded one unless a Meta Model API key is set (then Live calls are possible).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from headroom.agent.operator import DEMO_REQUESTS, SYSTEM, run
from headroom.agent.site import ReplaySite
from headroom.capacity import hosting_capacity
from headroom.config import DATA, LOCAL_TZ, ROOT
from headroom.meta.client import FailoverClient, ModelClient, RecordingMissing, RoleConfig, transcribe
from headroom.tariff import load_gold

st.set_page_config(page_title="Headroom", page_icon=str(ROOT / "assets" / "icon.png"), layout="wide")
RESULTS = DATA / "results"
VOICE_WAV = DATA / "voice" / "question.wav"  # the recorded spoken request (scripts/record_voice.py)
NAMES = {"headroom": "Headroom", "llf": "Standard smart charging", "static_cap": "Static cap"}
COLORS = {"headroom": "#5eead4", "llf": "#f59e0b", "static_cap": "#94a3b8", "building": "#818cf8"}
TARIFF_NAMES = {"cpau-e-4-tou-2026": "Palo Alto E-4 TOU", "cpau-e-7-tou-2026": "Palo Alto E-7 TOU",
                "pge-b-19-2026": "PG&E B-19"}
DAYS = {"2017-06-21": "21 June 2017 · heat-wave alert", "2017-06-20": "20 June 2017 · heat-wave alert",
        "2017-06-14": "14 June 2017 · ordinary weekday"}
SPIKE_KW = 50

STYLE = """<style>
  /* the icon font, from Google Fonts: on the hosted app Streamlit's own copy was not registered, and then
     icon names showed as words */
  @font-face { font-family:"Material Symbols Rounded"; font-style:normal; font-weight:100 700; font-display:block;
    src:url(https://fonts.gstatic.com/s/materialsymbolsrounded/v384/sykg-zNym6YjUruM-QrEh7-nyTnjDwKNJ_190Fjzag.woff2) format("woff2"); }
  .block-container { padding-top:2.4rem; padding-bottom:4rem; max-width:1180px; }
  header[data-testid="stHeader"] { background:transparent; }
  [data-testid="stMetricValue"] { font-weight:600; letter-spacing:-.02em; }
  [data-testid="stMetricLabel"] p { color:#8b96aa; }
  [data-testid="stSidebarNav"] a span { font-size:14.5px; }
  .hr-eyebrow { color:#5eead4; font-size:12.5px; letter-spacing:.14em; text-transform:uppercase; font-weight:500; }
  .hr-title { font-size:clamp(28px,3.6vw,42px); font-weight:600; letter-spacing:-.025em; line-height:1.12; margin:.35rem 0 .55rem; color:#e8ecf4; }
  .hr-sub { color:#8b96aa; font-size:17px; line-height:1.6; max-width:780px; margin-bottom:.4rem; }
  .hr-h { font-size:20px; font-weight:600; letter-spacing:-.01em; margin:1.8rem 0 .2rem; color:#e8ecf4; }
  .hr-note { color:#8b96aa; font-size:13.5px; line-height:1.6; }
  .hr-grid { display:grid; gap:12px; }
  .hr-card { border:1px solid #1e293b; border-radius:16px; background:linear-gradient(180deg,#121b2d,#0d1424); padding:20px 22px; }
  .hr-card h4 { font-size:12.5px; color:#5eead4; letter-spacing:.12em; text-transform:uppercase; font-weight:500; margin:0 0 10px; }
  .hr-card b { color:#e8ecf4; } .hr-card p, .hr-card li { color:#cbd3e1; font-size:14.5px; line-height:1.55; }
  .hr-card ul { list-style:none; padding:0; margin:0; } .hr-card li { padding:7px 0; border-bottom:1px solid rgba(148,163,184,.08); }
  .hr-card li:last-child { border-bottom:0; } .hr-card li small { display:block; color:#8b96aa; font-size:12.5px; }
  .hr-io { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
  .hr-steps { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
  .hr-step .n { width:30px; height:30px; border-radius:99px; display:grid; place-items:center; font-size:13px; font-weight:600;
                color:#5eead4; border:1px solid rgba(94,234,212,.35); background:rgba(45,212,191,.08); margin-bottom:10px; }
  .hr-step h5 { font-size:16px; font-weight:600; margin:0 0 4px; color:#e8ecf4; }
  .hr-pill { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:99px; font-size:13px; color:#cbd3e1;
             border:1px solid #1e293b; background:#111a2b; }
  .hr-pill i { width:8px; height:8px; border-radius:99px; display:inline-block; }
  .hr-big { font-size:56px; font-weight:600; letter-spacing:-.03em; line-height:1; }
  .hr-rise { animation:hrRise .55s cubic-bezier(.2,.7,.2,1) both; }
  @keyframes hrRise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
  .hr-bay { display:grid; grid-template-columns:132px 1fr; gap:14px; align-items:center; }
  .hr-bay img { width:132px; border-radius:10px; border:1px solid #1e293b; }
  .hr-chip { display:inline-block; padding:3px 10px; border-radius:99px; font-size:12.5px; font-weight:500; }
  .hr-chip.blocked { background:rgba(251,113,133,.14); color:#fb7185; border:1px solid rgba(251,113,133,.4); }
  .hr-chip.charging { background:rgba(45,212,191,.12); color:#5eead4; border:1px solid rgba(94,234,212,.4); }
  .hr-chip.free { background:rgba(148,163,184,.10); color:#cbd3e1; border:1px solid #1e293b; }
  .hr-chip.check { background:rgba(251,191,36,.12); color:#fbbf24; border:1px solid rgba(251,191,36,.4); }
  .hr-models { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .hr-model { border:1px solid #1e293b; border-radius:14px; background:#0e1626; padding:13px 15px; transition:opacity .35s, border-color .35s; }
  .hr-model .k { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11px; color:#64748b; letter-spacing:.04em; }
  .hr-model .t { font-size:15.5px; font-weight:600; color:#e8ecf4; margin:3px 0 2px; }
  .hr-model .d { font-size:12.5px; color:#8b96aa; line-height:1.45; }
  .hr-model .s { display:inline-flex; align-items:center; gap:6px; margin-top:9px; font-size:12px; font-weight:500; }
  .hr-model .s i { width:7px; height:7px; border-radius:99px; display:inline-block; background:currentColor; }
  .hr-model.up .s { color:#5eead4; } .hr-model.down { opacity:.45; } .hr-model.down .s { color:#fb7185; }
  .hr-model.standby .s { color:#94a3b8; } .hr-model.live { border-color:rgba(94,234,212,.55); box-shadow:0 0 0 1px rgba(94,234,212,.15) inset; }
  .hr-model.live .s { color:#5eead4; } .hr-model.live .s i { animation:hrPulse 1.4s ease-in-out infinite; }
  @keyframes hrPulse { 50% { opacity:.25; } }
  .hr-kicker { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:11.5px; color:#5eead4; letter-spacing:.06em; }
  @media (max-width:900px) { .hr-io, .hr-steps, .hr-models { grid-template-columns:1fr; } }
</style>"""


# ------------------------------------------------------------------------------------------ helpers

def hero(eyebrow: str, title: str, sub: str = "") -> None:
    st.html(f'<div class="hr-rise"><div class="hr-eyebrow">{eyebrow}</div><div class="hr-title">{title}</div>'
            + (f'<div class="hr-sub">{sub}</div>' if sub else "") + "</div>")


def heading(text: str) -> None:
    st.html(f'<div class="hr-h">{text}</div>')


def styled(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color="#8b96aa", size=12),
                      margin=dict(l=64, r=16, t=30, b=10), hoverlabel=dict(bgcolor="#111a2b", font_color="#e8ecf4"),
                      legend=dict(orientation="h", y=-0.18, font=dict(color="#cbd3e1")))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False, linecolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.08)", zeroline=False)
    return fig


def image_uri(img: Image.Image) -> str:
    import base64
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def md(text: str) -> str:
    """Show $ amounts as money, not as the start of a math formula (Streamlit markdown reads $...$ as LaTeX)."""
    return text.replace("$", r"\$")


# ------------------------------------------------------------------------------------------ data

@st.cache_resource(show_spinner="Simulating the month on real charging sessions (about a minute, once)…")
def load_site(month: str, n_ports: int) -> ReplaySite:
    return ReplaySite(month=month, n_ports=n_ports)


@st.cache_data(show_spinner=False)
def read_tariff(doc: str) -> dict:
    """The recorded tariff reading, verified once per server: reading the PDF and re-billing a year is the
    slowest step in the app, and its answer never changes for a given document."""
    return load_site("2017-06", 24).fork().read_tariff_document(doc)


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
    out["spike_usd"] = SPIKE_KW * sum(d.usd_per_kw for d in summer.demand)
    out["spike_energy_usd"] = SPIKE_KW * 0.25 * max(r.usd_per_kwh for r in summer.energy)
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
    fig.add_vrect(x0=day0 + pd.Timedelta(hours=16), x1=day0 + pd.Timedelta(hours=21), fillcolor="#fbbf24",
                  opacity=0.07, line_width=0, annotation_text="4–9 pm: priciest demand window",
                  annotation_position="top left", annotation_font_color="#fbbf24")
    q = site.forecast[t_now]
    ahead = now + pd.to_timedelta(np.arange(q.shape[0]) * 15, unit="min")
    keep = ahead <= frame["time"].iloc[-1]
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ahead[keep], y=q[keep, 0], fill="tonexty", fillcolor="rgba(129,140,248,0.16)",
                             line=dict(width=0), name="Building forecast (likely range)"))
    past = frame[frame["time"] <= now]
    fig.add_trace(go.Scatter(x=past["time"], y=past["building"], name="Building alone", fill="tozeroy",
                             fillcolor="rgba(129,140,248,0.10)", line=dict(color=COLORS["building"], width=1.6)))
    for name in (["static_cap"] if show_static else []) + ["llf", "headroom"]:
        label = "Site with Headroom" if name == "headroom" else f"Site with {NAMES[name].lower()}"
        fig.add_trace(go.Scatter(x=past["time"], y=past[name], name=label,
                                 line=dict(color=COLORS[name], width=3.2 if name == "headroom" else 2)))
    fig.add_hline(y=site.scenario.service_kw, line_dash="dot", line_color="#fb7185",
                  annotation_text="Grid connection limit", annotation_position="bottom right",
                  annotation_font_color="#fb7185")
    fig.add_vline(x=now, line_color="rgba(232,236,244,.5)", line_width=1)
    styled(fig, 430)
    fig.update_layout(yaxis_title="kW", hovermode="x unified")
    fig.update_yaxes(range=[450, 1050])
    return fig


def speak(text: str) -> None:
    """Read the answer aloud with the browser's built-in voice (not a model call)."""
    import streamlit.components.v1 as components  # a zero-height frame: works on every Streamlit version

    safe = json.dumps(text[:600])
    components.html(f"<script>const s=window.parent.speechSynthesis||window.speechSynthesis;"
                    f"const u=new SpeechSynthesisUtterance({safe});u.rate=1.05;s.cancel();s.speak(u);</script>", height=0)


@st.cache_data
def live_available() -> bool:
    """Live calls need a Meta Model API key in .env; without one (as on the public link) the app only replays."""
    return bool(RoleConfig.from_env("spark").api_key)


def replay_mode() -> bool:
    return not live_available() or (st.session_state.get("calls") or "Replay") == "Replay"


def current_site() -> ReplaySite:
    """This visitor's own copy of the simulated month: one visitor's questions never change another's site."""
    month = st.session_state.get("day", "2017-06-21")[:7]
    if f"site_{month}" not in st.session_state:
        st.session_state[f"site_{month}"] = load_site(month, 24).fork()
    return st.session_state[f"site_{month}"]


# ------------------------------------------------------------------------------------------ pages

def start() -> None:
    h = headline()
    hero("AI operator for EV charging", "More EV chargers on the grid connection you already have.",
         "Headroom watches the building, charges the cars in the room it leaves, and guarantees the site never goes over "
         "its limit. Meta's Muse Spark runs it day to day: the site manager just asks.")
    if "usd_headroom" in h:
        cols = st.columns(4)
        cols[0].metric("Chargers that fit", f"{h['chargers_headroom']}",
                       f"+{h['chargers_headroom'] - h['chargers_standard']} vs {h['chargers_standard']} standard",
                       help="On the same 1,000 kW connection, every driver served: Headroom vs standard smart charging")
        cols[1].metric("Demand charges", f"${h['usd_headroom']:,.0f}/mo", f"-{h['saving']:.0%} vs standard",
                       delta_color="inverse", help="Demand charges added by charging, per month, at 24 chargers")
        cols[2].metric("Drivers' energy delivered", f"{h['served']:.0%}")
        cols[3].metric("Times over the grid limit", f"{h['violations']}")
        st.caption("Held-out months (May–Oct 2017), real Palo Alto drivers, a real metered building, the utility's own tariff.")

    heading("How a site uses it")
    st.html(f"""<div class="hr-io hr-rise">
      <div class="hr-card"><h4>You give</h4><ul>
        <li><b>A year of meter data</b><small>the building's export from its utility (Green Button)</small></li>
        <li><b>The utility tariff</b><small>the PDF; Muse Spark reads it and checks it reproduces the bill</small></li>
        <li><b>Connection size and chargers</b><small>for example 1,000 kW and the chargers you want</small></li></ul></div>
      <div class="hr-card"><h4>Headroom does</h4><ul>
        <li><b>Simulates real drivers at your building</b><small>months of charging, before you buy anything</small></li>
        <li><b>Plans every 15 minutes once installed</b><small>charges in the room the building leaves</small></li>
        <li><b>Guarantees the limit and drives the chargers</b><small>a safety layer, then OCPP power limits</small></li></ul></div>
      <div class="hr-card"><h4>You get</h4><ul>
        <li><b>How many chargers fit</b><small>here {h.get('chargers_headroom', '–')} instead of {h.get('chargers_standard', '–')}</small></li>
        <li><b>Every driver fully charged, a lower bill</b><small>{h.get('saving', 0):.0%} lower demand charges, 0 times over the limit</small></li>
        <li><b>Checked answers to any question</b><small>"6 vans need 40 kWh by 6 am. Can we do it?"</small></li></ul></div>
    </div>""")

    heading("Try it")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.markdown("**Check your building**")
        st.caption("Upload a meter file, or use a real sample building, and see how many chargers fit.")
        st.page_link(PAGES["check"], label="Open Check", icon=":material/apartment:")
    with b.container(border=True):
        st.markdown("**Watch it run**")
        st.caption("Press play on a real heat-wave day and watch every 15-minute decision.")
        st.page_link(PAGES["replay"], label="Open the replay", icon=":material/play_circle:")
    with c.container(border=True):
        st.markdown("**Ask the operator**")
        st.caption("Muse Spark's real recorded answers, each number checked. Cut the internet to see Glimmer.")
        st.page_link(PAGES["operator"], label="Open the operator", icon=":material/forum:")

    heading("Every 15 minutes, four steps")
    st.html("""<div class="hr-steps hr-rise">
      <div class="hr-card hr-step"><div class="n">1</div><h5>See the building</h5><p>A 24-hour forecast with a calibrated worst case.</p></div>
      <div class="hr-card hr-step"><div class="n">2</div><h5>Plan the charging</h5><p>A linear program fills the room under the peak already paid for.</p></div>
      <div class="hr-card hr-step"><div class="n">3</div><h5>Guarantee the limit</h5><p>A safety layer nothing above it can override.</p></div>
      <div class="hr-card hr-step"><div class="n">4</div><h5>Send to the chargers</h5><p>OCPP power limits that expire, so control fails safe.</p></div>
    </div>""")


def replay() -> None:
    hero("Watch it run", "A real heat-wave day, every 15 minutes.",
         "Palo Alto drivers at 24 chargers (4× today's garage), a UC Berkeley building's metered load and Palo Alto's "
         "E-4 tariff, on the CAISO Flex Alert of 21 June 2017. Press play to watch Headroom decide.")
    top = st.columns([2, 3])
    day = top[0].selectbox("Day", list(DAYS), format_func=DAYS.get, index=list(DAYS).index(st.session_state["day"]))
    if day != st.session_state["day"]:
        st.session_state["day"] = day
        st.session_state.pop("replay_t", None)
    site = current_site()
    times = list(pd.date_range(f"{day} 00:00", f"{day} 23:45", freq="15min"))
    if "replay_t" not in st.session_state or st.session_state["replay_t"] not in times:
        at = pd.Timestamp(f"{day} {st.query_params.get('at', '14:00')}").floor("15min")  # ?at=HH:MM opens there
        st.session_state["replay_t"] = at if at in times else times[56]
    playing = st.session_state.get("playing", False)
    with top[1]:
        if st.button("Pause" if playing else "Play the day", icon=":material/pause:" if playing else ":material/play_arrow:",
                     type="primary"):
            st.session_state["playing"] = not playing
            if not playing and st.session_state["replay_t"] >= times[-1]:
                st.session_state["replay_t"] = times[0]
            st.rerun()
    show_static = st.toggle("Also show a fixed charging cap", value=False)

    @st.fragment(run_every=0.9 if playing else None)
    def day_view() -> None:
        if st.session_state.get("playing"):
            i = times.index(st.session_state["replay_t"])
            if i + 1 < len(times):
                st.session_state["replay_t"] = times[i + 1]
            else:
                st.session_state["playing"] = False
        now = st.select_slider("Time of day", options=times, key="replay_t", format_func=lambda t: t.strftime("%H:%M"))
        site.now_local = now.strftime("%Y-%m-%dT%H:%M")
        t_now = site.t
        bills = month_to_date(site, t_now)
        m = st.columns(4)
        m[0].metric("Building now", f"{site.scenario.building_kw[t_now - 1]:,.0f} kW")
        m[1].metric("Cars plugged in", int(((site.scenario.arrive <= t_now) & (site.scenario.depart > t_now)).sum()))
        m[2].metric("Demand charges this month", f"${bills['headroom']:,.0f}",
                    f"-${bills['llf'] - bills['headroom']:,.0f} vs standard", delta_color="inverse",
                    help="Demand charges added by charging so far this month, Headroom vs standard smart charging")
        site_kw = site.scenario.building_kw[:t_now] + site.results["headroom"].ev_kw[:t_now]
        m[3].metric("Grid-limit violations", int((site_kw > site.scenario.service_kw + 1e-6).sum()),
                    help="15-minute intervals this month so far with the site above its grid connection")
        st.plotly_chart(replay_chart(site, day_frame(site, day), now, t_now, show_static), theme=None, config={"displayModeBar": False})

    day_view()
    st.html('<div class="hr-note"><b style="color:#818cf8">Indigo</b>: the building on its own, and its forecast ahead '
            '(the shaded band). <b style="color:#f59e0b">Amber</b>: the same cars with standard smart charging, which '
            'can\'t see the building, so it charges into the evening peak. <b style="color:#5eead4">Teal</b>: Headroom, '
            'which fills the quiet hours and stays under the peak the month has already paid for.</div>')


def model_strip(link_up: bool) -> str:
    """The three Meta models behind the operator, and which of them can answer right now."""
    def card(key: str, title: str, what: str, state: str, label: str) -> str:
        return (f'<div class="hr-model {state}"><div class="k">{key}</div><div class="t">{title}</div>'
                f'<div class="d">{what}</div><div class="s"><i></i>{label}</div></div>')
    cut = "unreachable: internet cut"
    return ('<div class="hr-models hr-rise">'
            + card("muse-spark-1.3 · cloud", "Muse Spark", "Answers, and acts only through checked tools.",
                   "live" if link_up else "down", "answering" if link_up else cut)
            + card("muse-voice-transcribe-1.0 · cloud", "Muse Voice Transcribe", "Turns a spoken request into text.",
                   "up" if link_up else "down", "reachable" if link_up else cut)
            + card("muse-glimmer · on site", "Muse Glimmer", "Runs on the site's own GPU; takes over when the link drops.",
                   "standby" if link_up else "live", "standby" if link_up else "answering on site")
            + "</div>")


def voice_request(client: FailoverClient, link_up: bool, shown: list[dict]) -> None:
    """The recorded spoken request, run the way a live one is: transcribed, then answered through the tools."""
    from headroom.agent.operator import VOICE_REQUEST_AT

    if not link_up:
        shown.append({"role": "assistant", "text": "Voice needs the internet: Muse Voice Transcribe runs in the cloud. "
                      "Type or tap a question instead and Muse Glimmer answers on site. Charging carries on unchanged."})
        return
    with st.status("Muse Voice Transcribe is listening…", expanded=True) as status:
        text = transcribe(VOICE_WAV.read_bytes(), mode="replay" if replay_mode() else "auto")
        time.sleep(0.6)
        st.write(f"🎤 Heard: “{text}”")
        status.update(label="Muse Spark is checking with the tools…")
        site = load_site("2017-06", 24).fork()  # the request was recorded on a fresh site at 6 pm
        site.now_local = VOICE_REQUEST_AT
        turn = run(site, text, history=[{"role": "system", "content": SYSTEM}], client=client)
        for step in turn.steps:
            if step.kind == "tool":
                time.sleep(0.45)
                st.write(f"{'✅' if step.ok else '❌'} `{step.tool}`")
        status.update(label="Answered", state="complete", expanded=False)
    check = ("every number traced to a tool result ✅" if not turn.unsupported else
             "⚠️ not found in any tool result: " + ", ".join(turn.unsupported))
    shown.append({"role": "user", "text": "🎤 " + text, "note": "Spoken, then transcribed by Muse Voice Transcribe", "voice": True})
    shown.append({"role": "assistant", "text": turn.answer, "note": f"Muse Spark (cloud) · {check}",
                  "steps": [{"tool": s.tool, "result": s.result, "ok": s.ok} for s in turn.steps if s.kind == "tool"]})
    speak(turn.answer)


def operator() -> None:
    hero("Ask the operator", "The site manager just asks. Meta's Muse Spark answers.",
         "It can only act through checked tools, never the chargers, and every number it says is traced to a tool "
         "result. Ask by tapping, typing or speaking, then cut the internet and watch Muse Glimmer take over on site.")
    left, right = st.columns([5, 1.3], vertical_alignment="center")
    with right:
        down = st.toggle("Cut the internet", value=st.session_state.get("link_down", False), key="link_toggle",
                         help="The cloud models become unreachable and Muse Glimmer answers from the site's own GPU. "
                              "Charging never depends on any model.")
    if replay_mode() and down != st.session_state.get("link_down", False):
        # each model's recorded conversation replays from its own start: a mixed history has no recording
        st.session_state["history"], st.session_state["shown"] = [{"role": "system", "content": SYSTEM}], []
        st.session_state["model_switched"] = True
    st.session_state["link_down"] = down
    link_up = not down
    with left:
        st.html(model_strip(link_up))
    st.html('<div class="hr-note" style="margin:.35rem 0 1rem">The planner and the safety layer run on site and never '
            'wait for a model: with every model gone, the chargers keep their limits.</div>')
    if st.session_state.get("model_switched") and not st.session_state.get("shown"):
        st.info(("Internet cut: Muse Glimmer now answers on site. " if down else "Back online: Muse Spark answers again. ")
                + "The recorded conversation starts over so each model's real answers replay in order.",
                icon=":material/wifi_off:" if down else ":material/wifi:")
    site = current_site()
    client = FailoverClient("spark", "glimmer", mode="replay" if replay_mode() else "auto", link_up=link_up)
    history = st.session_state.setdefault("history", [{"role": "system", "content": SYSTEM}])
    shown = st.session_state.setdefault("shown", [])
    for msg in shown:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/bolt:"):
            st.markdown(md(msg["text"]))
            if msg.get("note"):
                st.caption(msg["note"])
            if msg.get("steps"):
                with st.expander(f"How it got this: {len(msg['steps'])} tool call(s)"):
                    for step in msg["steps"]:
                        st.markdown(f"{'✅' if step['ok'] else '❌ rejected'} `{step['tool']}`")
                        st.json(step["result"], expanded=False)

    asked = sum(m["role"] == "user" and not m.get("voice") for m in shown)  # the recorded questions asked
    request, ask_at = None, None
    if replay_mode() and asked < len(DEMO_REQUESTS) and st.session_state.get("day", "2017-06-21") == "2017-06-21":
        ask_at, text = DEMO_REQUESTS[asked]
        if st.button(f"Ask: “{text}”", type="primary", icon=":material/forum:"):
            request = text
        else:
            ask_at = None
        st.caption(f"Recorded conversation, question {asked + 1} of {len(DEMO_REQUESTS)}: the models' real answers, replayed.")
    elif replay_mode() and asked < len(DEMO_REQUESTS):
        st.caption("The recorded conversation is on 21 June 2017: choose that day on *Watch it run*.")
    with st.container(border=True):
        st.html('<div class="hr-kicker">ASK BY VOICE</div>')
        a, b = st.columns([3, 1.4], vertical_alignment="center")
        with a:
            st.audio(str(VOICE_WAV))
            st.caption("A recorded spoken request, voiced by a built-in system synthesiser. Muse Voice Transcribe "
                       "turns it into text, then Muse Spark answers it.")
        with b:
            send_voice = st.button("Send this voice request", icon=":material/mic:", use_container_width=True)
        voice = None
        if not replay_mode():
            voice = st.audio_input("Or record your own", help="Muse Voice Transcribe turns it into text; Muse Spark answers")
    if send_voice:
        try:
            voice_request(client, link_up, shown)
        except RecordingMissing as exc:
            st.warning(str(exc))
        else:
            st.rerun()
    if shown and st.button("Start over", icon=":material/restart_alt:"):
        st.session_state["history"], st.session_state["shown"] = [{"role": "system", "content": SYSTEM}], []
        for key in [k for k in st.session_state if k.startswith("site_")]:
            del st.session_state[key]  # a fresh copy of the month for this visitor only
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
        with st.spinner("Checking with the tools…"):
            turn = run(site, request, history=history, client=client)
        st.session_state["history"] = turn.messages
        steps = [{"tool": s.tool, "result": s.result, "ok": s.ok} for s in turn.steps if s.kind == "tool"]
        who = "Muse Glimmer (on site)" if turn.answered_by == "glimmer" else "Muse Spark (cloud)"
        check = ("every number traced to a tool result ✅" if not turn.unsupported else
                 "⚠️ not found in any tool result: " + ", ".join(turn.unsupported))
        shown.append({"role": "assistant", "text": turn.answer, "note": f"{who} · {check}", "steps": steps})
        if spoken:  # voice in, voice out: typed and tapped questions stay silent
            speak(turn.answer)
    except RecordingMissing:
        shown.append({"role": "assistant", "text": "There's no recorded answer for that question: the recorded "
                      "conversation is the question buttons above. " + (
                          "Switch **Model calls** to Live in the sidebar to ask anything." if live_available() else
                          "This public demo replays recorded answers only.")})
    except ConnectionError:
        shown.append({"role": "assistant", "text": "No model is reachable right now. Charging carries on "
                      "unchanged: the planner and safety layer never depend on a model."})
    except RuntimeError as exc:
        shown.append({"role": "assistant", "text": str(exc)})
    st.rerun()


def camera() -> None:
    from headroom.agent.bays import decide, load_bays, observe, occupancy_accuracy

    hero("What the camera sees", "A meter can't see a parked car that isn't charging. The camera can.",
         "In the real Palo Alto data, cars sit plugged in but finished for 28.2% of the time. Muse Spark watches the bay "
         "camera; a bay counts as blocked only if the camera sees a car <b>and</b> the charger reports no power.")
    video = DATA / "video" / "lot_close_5fps.mp4"
    left, right = st.columns([3, 2])
    with left:
        # a static file (static/, server.enableStaticServing): a media-manager URL can vanish mid-load on a rerun
        st.html('<video src="app/static/lot_close_5fps.mp4" controls muted playsinline preload="metadata" '
                'style="width:100%;border-radius:14px;border:1px solid #1e293b"></video>')
        st.caption("Public-domain parking-lot footage, 63 seconds; six bays marked as EV bays for illustration.")
    with right:
        st.html("""<div class="hr-card"><h4>What happens when you press the button</h4><ul>
          <li><b>1 · The clip goes to Muse Spark</b><small>with a still of the bays, numbered</small></li>
          <li><b>2 · It reports every bay</b><small>occupied, the vehicle, whether a cable is visible</small></li>
          <li><b>3 · The chargers report their power</b><small>straight from the charger network</small></li>
          <li><b>4 · Both must agree</b><small>a car, and no power flowing: then the bay is blocked</small></li></ul></div>""")
        go_ = st.button("Run the camera check", type="primary", icon=":material/videocam:", width="stretch")
    if not go_:
        return
    bays = load_bays("lot_close")
    stage = st.container()
    pause = lambda s: time.sleep(s)  # noqa: E731 - the stages appear one after another
    with stage:
        heading("1 · What Muse Spark receives")
        frames = sorted((DATA / "video" / "frames").glob("frame*.jpg"))
        cols = st.columns(len(frames) + 1)
        for col, f in zip(cols, frames):
            col.image(str(f), caption="a frame of the clip")
        cols[-1].image(str(DATA / "video" / "lot_close.bays.png"), caption="the still, bays numbered")
        pause(1.0)
        try:
            with st.spinner("Muse Spark is watching the clip…"):
                report = observe(video, DATA / "video" / "lot_close.jpg", bays,
                                 client=ModelClient("spark", mode="replay" if replay_mode() else "auto"))
        except (RecordingMissing, RuntimeError) as exc:
            st.warning(str(exc))
            return
        charger_kw = {b.number: 0.0 for b in bays}
        charger_kw[3] = 6.1  # the charger on bay 3 reports a car charging
        rows = decide(report, charger_kw)
        still = Image.open(DATA / "video" / "lot_close.bays.png")
        heading("2 · What it sees, bay by bay · 3 · what each charger reports · 4 · the decision")
        grid = st.columns(3)
        for k, (row, bay) in enumerate(zip(rows, bays)):
            cam = row["camera"]
            x0, y0, x1, y1 = bay.box
            crop = still.crop((max(0, x0 - 10), max(0, y0 - 40), x1 + 10, y1 + 10))
            with grid[k % 3].container(border=True):
                st.html(f'<img src="{image_uri(crop)}" style="width:100%;border-radius:10px;border:1px solid #1e293b">')
                st.markdown(f"**Bay {row['bay']}** · camera: {'a ' + cam['vehicle'] if cam['occupied'] else 'empty'}"
                            f"{', cable ' + cam['plugged_in'] if cam['occupied'] else ''}")
                st.markdown(f"Charger reports **{row['charger_kw']:.1f} kW**")
                st.html(f'<span class="hr-chip {row["status"]}">{row["status"]}</span>'
                        + (f' <span class="hr-note">{row["action"]}</span>' if row["action"] != "none" else ""))
            pause(0.45)
        acc = occupancy_accuracy(report, "lot_close")
        blocked = sum(r["status"] == "blocked" for r in rows)
        st.success(f"{blocked} blocked bays found and taken out of the charging plan · camera calls match hand labels "
                   f"for {acc['correct']} of {acc['bays']} bays · Muse Spark's real recorded answer", icon=":material/check_circle:")


def check_building() -> None:
    from headroom.check import check_meter, load_meter

    h = headline()
    hero("Check your building", "How many chargers fit on your existing connection?",
         "Upload a year of your building's meter data. Headroom simulates real Palo Alto drivers on it, month by month, "
         "with standard smart charging and with Headroom, and tells you how many chargers each can host with every driver "
         "charged and the site never over its limit.")
    sample = DATA / "samples" / "sample_building_green_button.csv"
    a, b = st.columns([3, 2])
    with a.container(border=True):
        st.markdown("**1 · Your meter data**")
        upload = st.file_uploader("Meter data (CSV)", type=["csv"], label_visibility="collapsed",
                                  help="Green Button export from the utility, or any table with a timestamp and kW or kWh: "
                                       "5-, 15-, 30-minute or hourly readings, at least four months")
        use_sample = sample.exists() and st.button("Or try a real sample building", icon=":material/apartment:")
    with b.container(border=True):
        st.markdown("**2 · Your grid connection**")
        service = st.number_input("Connection size (kW)", min_value=0, value=0, step=50,
                                  help="Leave at 0 to estimate it from your peak ÷ 0.8 (the electrical-code rule)")
        st.caption("Leave at 0 if you don't know it: Headroom estimates it from your peak, as the electrical code does.")
    raw = upload.getvalue() if upload is not None else (sample.read_bytes() if use_sample else None)
    if raw is None:
        st.caption("Your file is used only for the calculation and isn't stored. A check simulates months of charging at "
                   "many fleet sizes: a few minutes, longer for a large building. The sample answers instantly.")
        context(h)
        return
    # line endings don't change the data: Git checks the sample out with CRLF on Windows and LF on Linux hosts
    key = hashlib.sha256(raw.replace(b"\r\n", b"\n") + str(service).encode()).hexdigest()[:16]
    cache = RESULTS / "check_cache" / f"{key}.json"
    status = None
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
        except ValueError as exc:
            if status is not None:
                status.update(label="Stopped", state="error", expanded=False)
            st.error(f"Can't check this building: {exc}.")
            return
        except (StopIteration, KeyError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("Couldn't read that file. It needs a timestamp column and a kW or kWh column "
                     "(a Green Button CSV export works).")
            return
    at_least = lambda key: "≥ " if answer.get(key) else ""  # noqa: E731 - the search stopped at its limit
    std, hr = answer["chargers_standard_smart_charging"], answer["chargers_headroom"]
    heading("3 · Your answer")
    with st.container(border=True):
        st.html(f"""<div class="hr-rise" style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
          <div><div class="hr-note">With standard smart charging</div><div class="hr-big" style="color:#f59e0b">{at_least('standard_at_least')}{std}</div>
            <div class="hr-note">chargers</div></div>
          <div><div class="hr-note">With Headroom, on the same connection</div><div class="hr-big" style="color:#5eead4">{at_least('headroom_at_least')}{hr}</div>
            <div class="hr-note">chargers · {hr - std:+d}, every driver charged, {answer.get('violations_at_reported_sizes', answer['service_violations'])} times over the limit</div></div>
        </div>""")
        fig = go.Figure(go.Bar(x=[std, hr], y=["Standard smart charging", "Headroom"], orientation="h",
                               marker_color=[COLORS["llf"], COLORS["headroom"]], text=[std, hr], textposition="outside"))
        styled(fig, 150).update_layout(margin=dict(l=10, r=48, t=6, b=6), showlegend=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        usd = answer.get("added_demand_usd_per_month_at_standard_capacity") or {}
        if "llf" in usd and "headroom" in usd:
            st.markdown(md(f"At {std} chargers, charging adds **${usd['headroom']:,}/month** in demand charges with "
                           f"Headroom, versus ${usd['llf']:,} with standard smart charging."))
        if "buildings" in h:
            st.markdown(f"For comparison: across the {h['buildings']} buildings tested, the median gain is "
                        f"**{h['building_gain']:+.0%}**."
                        + (" This sample (a real office meter from BDG2) is one of the largest gains in that study."
                           if use_sample and upload is None else ""))
        st.caption(f"Screened {answer.get('your_months', ', '.join(answer['months']))} · building peak "
                   f"{answer['building_peak_kw']:,} kW · connection {answer['service_kw']:,.0f} kW"
                   f"{' (estimated: peak ÷ 0.8)' if answer['service_assumed_from_peak'] else ''} · drivers must get "
                   f"≥ 99% of their energy in every month · E-4 TOU tariff · real Palo Alto driver sessions"
                   + (f" · safety allowance for a sudden load rise: {answer['ramp_allowance_kw']:,.0f} kW, from the "
                      f"building's past year" if answer.get("ramp_allowance_kw") else ""))
    context(h)


def context(h: dict) -> None:
    if "buildings" not in h:
        return
    with st.expander(f"Tested on {h['buildings']} real buildings at {h['sites']} US sites"):
        ps = pd.read_csv(RESULTS / "portfolio_summary.csv")
        a, b, c = st.columns(3)
        a.metric("Median extra chargers, same connection", f"{h['building_gain']:+.0%}")
        b.metric("Buildings where Headroom fits more", f"{h['buildings_more']:.0%}")
        c.metric("Grid-limit violations at these sizes", int(ps["violations_at_reported_sizes"].sum()))
        fig = go.Figure(go.Scatter(x=ps["chargers_standard_smart_charging"], y=ps["chargers_headroom"], mode="markers",
                                   text=ps["primaryspaceusage"].fillna("") + " · " + ps["site_id"],
                                   marker=dict(color=COLORS["headroom"], size=9, opacity=0.8)))
        top = float(max(ps["chargers_headroom"].max(), ps["chargers_standard_smart_charging"].max()) + 2)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", hoverinfo="skip", line=dict(color="#475569", dash="dot")))
        styled(fig, 320).update_layout(showlegend=False, xaxis_title="Chargers: standard smart charging",
                                       yaxis_title="Chargers: Headroom")
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
        st.caption("Each dot is one real metered building (BDG2), screened on Jul–Aug 2017. Above the dotted line, "
                   "Headroom fits more. Buildings whose peak is brief gained a median +82%; with a tighter connection "
                   "(peak ÷ 0.9) the median gain rises to +31%. Method: scripts/portfolio_study.py")


def verified() -> None:
    from headroom.agent.tariff_reader import bill_check, field_accuracy

    hero("How the AI is checked", "Nothing Muse Spark produces is used until a check passes.",
         "It reads the utility's tariff into structured rates; they're accepted only if they reproduce the bill for every "
         "month of real metered load. Its answers are traced number by number, and the camera check needs the charger "
         "to agree.")
    site = current_site()
    with st.container(border=True):
        st.markdown("**Reading the utility's tariff**")
        doc = st.segmented_control("Tariff", list(TARIFF_NAMES), default="cpau-e-4-tou-2026",
                                   format_func=TARIFF_NAMES.get) or "cpau-e-4-tou-2026"
        if st.button("Read and verify", type="primary", icon=":material/fact_check:"):
            try:
                with st.spinner("Muse Spark is reading the tariff…"):
                    out = read_tariff(doc) if replay_mode() else site.read_tariff_document(doc)
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
            gold = load_gold("e-4-tou_2026")
            bad = gold.model_copy(deep=True)
            bad.seasons[0].demand[0].usd_per_kw = 10.26  # only the commodity column: a classic misread
            acc, bills = field_accuracy(bad, gold), bill_check(bad, gold, site.site["building_kw"])
            st.markdown(md(f"Palo Alto E-4, reading only one column of the summer peak demand charge ($10.26 instead of "
                           f"${gold.seasons[0].demand[0].usd_per_kw}): **{acc['correct']} of {acc['fields']} fields right, "
                           f"but the bill is off by {bills['worst_error']:.1%} in {bills['worst_month']}, so it's rejected.**"))
    st.page_link(PAGES["camera"], label="See the camera check", icon=":material/videocam:")


def proof() -> None:
    h = headline()
    hero("Proof and data", "Measured on months nothing was tuned on.",
         "Held-out months (May–Oct 2017) and real data only. Each table is a file in data/results/ produced by a script "
         "in scripts/.")
    bench, capacity, calibration, chargers, calls, data = st.tabs(
        ["Benchmark", "Charger capacity", "Forecast calibration", "Charger control (OCPP)", "Model call log",
         "Data and assumptions"])
    with chargers:
        demo = RESULTS / "ocpp_demo.json"
        if demo.exists():
            o = json.loads(demo.read_text())
            a, b, c = st.columns(3)
            a.metric("Commands accepted", f"{o['accepted']} of {o['messages']}")
            b.metric("Planned vs received limit", f"{o['max_limit_mismatch_w']:.0f} W apart")
            c.metric("All chargers updated in", f"{o['median_round_trip_ms_all_chargers']:.0f} ms")
            st.caption(f"Headroom's real decisions for {o['day']}, {o['window_local']} ({o['intervals']} intervals) "
                       f"sent to {o['chargers']} simulated chargers as OCPP 1.6J SetChargingProfile commands over "
                       "websockets. Each limit lasts one interval, so chargers fall back to their safe default if "
                       "the controller goes quiet. scripts/ocpp_demo.py")
            with st.expander("One command, as sent"):
                st.json(o["example_set_charging_profile"])
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
            other = RESULTS / "final_summary_pge-b-19_2026.csv"
            if other.exists():
                o = pd.read_csv(other, index_col="controller")
                b = o["added_demand_usd"]
                st.markdown(md(f"**Same site on PG&E's B-19 tariff:** Headroom ${b['headroom']:,.0f}/month vs "
                               f"${b['llf']:,.0f} for standard smart charging ({b['headroom'] / b['llf'] - 1:+.0%}), "
                               f"{o.loc['headroom', 'energy_served']:.1%} of energy delivered, "
                               f"{o.loc['headroom', 'service_violations']:.0f} violations."))
    with capacity:
        cap = RESULTS / "capacity_summary_test.csv"
        if cap.exists():
            summary = pd.read_csv(cap)
            fig = go.Figure()
            for name in ["static_cap", "llf", "headroom"]:
                s = summary[summary["controller"] == name]
                fig.add_trace(go.Scatter(x=s["n_ports"], y=s["months_drivers_served"], name=NAMES[name],
                                         mode="lines+markers", line=dict(color=COLORS[name], width=2.6)))
            styled(fig, 340).update_layout(yaxis_title="Share of months every driver was served", xaxis_title="Chargers",
                                           yaxis_tickformat=".0%")
            st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
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
            "- **Tariffs:** City of Palo Alto Utilities E-4 TOU and E-7 TOU (2026) and PG&E B-19 (2026), hand-checked.\n"
            "- **Grid alerts:** CAISO Flex Alert dates.\n"
            "- **Assumptions:** the building and the garage share a 1,000 kW connection (the two datasets come from "
            "neighbouring cities); 2026 prices on 2016–17 behaviour.\n"
            "- Full detail: docs/DATA_SPEC.md and docs/FORECASTING.md.")


# ------------------------------------------------------------------------------------------ app

PAGES = {
    "start": st.Page(start, title="Start", icon=":material/home:", url_path="start", default=True),
    "replay": st.Page(replay, title="Watch it run", icon=":material/play_circle:", url_path="replay"),
    "operator": st.Page(operator, title="Ask the operator", icon=":material/forum:", url_path="operator"),
    "camera": st.Page(camera, title="What the camera sees", icon=":material/videocam:", url_path="camera"),
    "check": st.Page(check_building, title="Check your building", icon=":material/apartment:", url_path="check_building"),
    "verified": st.Page(verified, title="How the AI is checked", icon=":material/verified:", url_path="verified"),
    "proof": st.Page(proof, title="Proof and data", icon=":material/analytics:", url_path="proof"),
}

st.html(STYLE)
st.logo(str(ROOT / "assets" / "logo.png"), icon_image=str(ROOT / "assets" / "icon.png"), size="large",
        link="https://headroom-meta.vercel.app/")
page = st.navigation({"": [PAGES["start"]],
                      "See it work": [PAGES["replay"], PAGES["operator"], PAGES["camera"]],
                      "Use it": [PAGES["check"]],
                      "Trust": [PAGES["verified"], PAGES["proof"]]})
st.session_state.setdefault("day", "2017-06-21")
with st.sidebar:
    if st.session_state.get("link_toggle", st.session_state.get("link_down")):  # the toggle's value this run
        st.html('<span class="hr-pill"><i style="background:#fb7185"></i>Internet cut · Muse Glimmer on site</span>')
    if live_available():
        st.segmented_control("Model calls", ["Replay", "Live"], default="Replay", key="calls",
                             help="Replay uses the recorded model responses; Live calls the models (billed)")
    else:
        st.caption("Replaying the models' real recorded answers.")
    st.caption("Built on Meta's Muse Spark 1.3 · Muse Voice Transcribe · Muse Glimmer")
page.run()
