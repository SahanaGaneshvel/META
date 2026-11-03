"""Build the demo video (1920x1080, 30 fps, captions burned in, no voice) from real material only.

    python scripts/video/capture.py   # screenshots of the real app -> video/shots/
    python scripts/video/build.py     # -> video/headroom_demo.mp4

Every number on screen comes from data/results/, the app, or the utility's own tariff document.
Music is added afterwards (docs/VIDEO.md); the video itself is silent.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio_ffmpeg  # noqa: E402
import pandas as pd  # noqa: E402

from headroom.config import DATA  # noqa: E402

W, H, FPS = 1920, 1080, 30
SHOTS = ROOT / "video" / "shots"
OUT = ROOT / "video"
CARDS = ROOT / "docs" / "cards"
BG, INK, MUTED, GREEN, ORANGE, BLUE = (11, 15, 20), (249, 250, 251), (156, 163, 175), (34, 197, 94), (217, 119, 6), (37, 99, 235)
FONT = "C:/Windows/Fonts/segoeui.ttf"
BOLD = "C:/Windows/Fonts/segoeuib.ttf"
SEMI = "C:/Windows/Fonts/seguisb.ttf"
FADE = 0.5  # seconds of crossfade between scenes


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@dataclass
class Scene:
    """A still (or a sequence of stills) shown for `seconds`, with a slow zoom and a caption."""
    frames: list[Image.Image]
    seconds: float
    caption: str = ""
    zoom: tuple[float, float] = (1.0, 1.04)
    focus: tuple[float, float] = (0.5, 0.5)  # where the zoom heads, as a share of width / height
    captions: list[tuple[float, str]] = field(default_factory=list)  # (start second, text) to change mid-scene


# ------------------------------------------------------------------ images

def app_shot(name: str, crop_sidebar: bool = True) -> Image.Image:
    """An app screenshot; without the sidebar (and scaled back to 1080p) unless the sidebar matters."""
    img = Image.open(SHOTS / f"{name}.png").convert("RGB")
    if crop_sidebar:
        img = img.crop((300, 40, 1920, 40 + 911)).resize((W, H), Image.LANCZOS)
    return img


def card(lines: list[tuple[str, str, int, tuple]], footer: str = "", y0: float | None = None) -> Image.Image:
    """Dark title card: lines of (text, font path, size, colour), centred."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    heights = [font(f, s).getbbox(t)[3] + s * 0.45 for t, f, s, _ in lines]
    y = y0 if y0 is not None else (H - sum(heights)) / 2
    for (text, path, size, colour), h in zip(lines, heights):
        f = font(path, size)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=colour)
        y += h
    if footer:
        f = font(FONT, 26)
        d.text(((W - d.textlength(footer, font=f)) / 2, H - 90), footer, font=f, fill=MUTED)
    return img


def tariff_page() -> Image.Image:
    """The real Palo Alto E-4 tariff, page 1, with the two demand charges circled."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(DATA / "docs" / "e-4-tou_effective_2026-07-01.pdf"))
    page = pdf[0]
    scale = 3.0
    img = page.render(scale=scale).to_pil().convert("RGB")
    text = page.get_textpage()
    height = page.get_height()
    d = ImageDraw.Draw(img)
    boxes = []
    for needle in ("30.18", "21.27"):
        hit = text.search(needle).get_next()
        if hit:
            first, count = hit
            rects = [text.get_charbox(i) for i in range(first, first + count)]
            l, b = min(r[0] for r in rects), min(r[1] for r in rects)
            r_, t = max(r[2] for r in rects), max(r[3] for r in rects)
            box = (l * scale - 18, (height - t) * scale - 14, r_ * scale + 18, (height - b) * scale + 14)
            d.rounded_rectangle(box, radius=14, outline=(220, 38, 38), width=7)
            boxes.append(box)
    # frame the part of the page with the rates, at 16:9
    cx = np.mean([(b[0] + b[2]) / 2 for b in boxes]) if boxes else img.width / 2
    cy = np.mean([(b[1] + b[3]) / 2 for b in boxes]) if boxes else img.height / 3
    w = min(img.width, 1700)
    h = w * 9 / 16
    left = int(np.clip(cx - w / 2, 0, img.width - w))
    top = int(np.clip(cy - h / 2, 0, img.height - h))
    return img.crop((left, top, left + int(w), top + int(h))).resize((W, H), Image.LANCZOS)


def meter_frames(seconds: float) -> list[Image.Image]:
    """The real building on the heat-wave day, with standard smart charging stacked on it, drawn hour by hour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from make_cards import heat_wave_day

    t, building, _, llf = heat_wave_day()
    hours = np.asarray(t.hour + t.minute / 60)
    site = building + llf
    window = (hours >= 16) & (hours < 21)
    peak_i = int(np.flatnonzero(window)[np.argmax(site[window])])
    frames = []
    n = int(seconds * FPS * 0.8)
    for k in range(n):
        upto = max(2, int(len(hours) * min(1, (k + 1) / (n * 0.85))))
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0b0f14")
        ax = fig.add_axes([0.07, 0.29, 0.88, 0.58], facecolor="#0b0f14")  # the caption sits below the axis
        ax.axvspan(16, 21, color="#facc15", alpha=0.10, lw=0)
        ax.text(18.5, 985, "4–9 pm: the priciest demand window", ha="center", va="top", color="#facc15", fontsize=17)
        ax.axhline(1000, color="#ef4444", ls=(0, (4, 4)), lw=1.5)
        ax.text(hours[0] + 0.1, 1003, "grid connection limit", color="#ef4444", fontsize=15, va="bottom")
        ax.plot(hours[:upto], building[:upto], color="#3b82f6", lw=3.2, label="the building, metered")
        ax.plot(hours[:upto], site[:upto], color="#f59e0b", lw=3.2, label="building + chargers, standard smart charging")
        ax.fill_between(hours[:upto], building[:upto], site[:upto], color="#f59e0b", alpha=0.25, lw=0)
        if upto > peak_i:  # the evening's highest 15 minutes with charging, and what charging added to it
            x = hours[peak_i]
            ax.annotate("", (x, site[peak_i]), xytext=(x, building[peak_i]),
                        arrowprops=dict(arrowstyle="<->", color="#ef4444", lw=2.5))
            ax.scatter([x], [site[peak_i]], s=200, color="#ef4444", zorder=5)
            ax.annotate(f"charging adds {site[peak_i] - building[peak_i]:,.0f} kW\nright at the evening peak",
                        (x, (site[peak_i] + building[peak_i]) / 2), xytext=(x - 7.2, site[peak_i] + 40),
                        color=INK_HEX, fontsize=22, arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2))
        ax.set_xlim(hours[0], hours[-1])
        ax.set_ylim(500, 1030)
        ax.set_xticks([6, 9, 12, 15, 18, 21], ["6 am", "9 am", "noon", "3 pm", "6 pm", "9 pm"])
        ax.tick_params(colors="#9ca3af", labelsize=17)
        ax.set_ylabel("kW", color="#9ca3af", fontsize=17)
        for side in ax.spines.values():
            side.set_color("#1f2937")
        ax.grid(color="#1f2937", lw=0.8)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=2, frameon=False, labelcolor=INK_HEX, fontsize=18)
        fig.text(0.95, 0.03, "Real metered load (BDG2) and real Palo Alto charging sessions · 21 June 2017, a CAISO "
                 "Flex Alert day", ha="right", color="#9ca3af", fontsize=14)
        fig.canvas.draw()
        frames.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
    return frames + [frames[-1]] * (int(seconds * FPS) - len(frames))


INK_HEX = "#f9fafb"


# ------------------------------------------------------------------ captions and motion

def draw_caption(img: Image.Image, text: str) -> Image.Image:
    if not text:
        return img
    f = font(SEMI, 46)
    d = ImageDraw.Draw(img)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if d.textlength(trial, font=f) > 1560 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    pad, gap = 30, 14
    widths = [d.textlength(x, font=f) for x in lines]
    box_w = max(widths) + 2 * pad
    box_h = len(lines) * 50 + (len(lines) - 1) * gap + 2 * pad
    x0, y0 = (W - box_w) / 2, H - 70 - box_h
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle((x0, y0, x0 + box_w, y0 + box_h), radius=22, fill=(11, 15, 20, 225))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    y = y0 + pad - 6
    for x, w in zip(lines, widths):
        d.text(((W - w) / 2, y), x, font=f, fill=INK)
        y += 50 + gap
    return img


def zoomed(img: Image.Image, z: float, focus: tuple[float, float]) -> Image.Image:
    if abs(z - 1) < 1e-4:
        return img
    w, h = W / z, H / z
    left = np.clip(focus[0] * W - w / 2, 0, W - w)
    top = np.clip(focus[1] * H - h / 2, 0, H - h)
    return img.resize((W, H), Image.BICUBIC, box=(left, top, left + w, top + h))


def render(scene: Scene):
    n = int(round(scene.seconds * FPS))
    timeline = sorted(scene.captions or [(0, scene.caption)])
    m = len(scene.frames)
    for i in range(n):
        pos = i * m / n
        k = min(m - 1, int(pos))
        src = scene.frames[k]
        a = (pos - k - 0.6) / 0.4  # the last 40% of each still's time blends into the next one
        if m > 1 and k + 1 < m and a > 0 and m < n / 3:
            src = Image.blend(src, scene.frames[k + 1], a)
        z = scene.zoom[0] + (scene.zoom[1] - scene.zoom[0]) * i / max(1, n - 1)
        text = [c for s, c in timeline if s <= i / FPS][-1]
        yield draw_caption(zoomed(src, z, scene.focus), text)


# ------------------------------------------------------------------ the story

def scenes() -> list[Scene]:
    final = pd.read_csv(DATA / "results" / "final_summary.csv", index_col="controller")
    saving = 1 - final.loc["headroom", "added_demand_usd"] / final.loc["llf", "added_demand_usd"]
    p = pd.read_csv(DATA / "results" / "portfolio_summary.csv")
    both = p[p["chargers_standard_smart_charging"] > 0]
    more = (both["extra_chargers"] > 0).mean()
    replay = [app_shot(f"replay_{h:02d}{m:02d}") for h in range(14, 22) for m in (0, 15, 30, 45) if (h, m) <= (21, 0)]
    hero = Image.open(CARDS / "hero_1080.png").convert("RGB")
    return [
        Scene([hero], 4.5, zoom=(1.0, 1.02)),
        Scene([card([("Offices, malls and depots want more EV chargers.", SEMI, 64, INK),
                      ("But the chargers share the building's grid connection.", FONT, 46, MUTED)])], 5.5,
              zoom=(1.0, 1.0)),
        Scene([tariff_page()], 9, zoom=(1.0, 1.08), captions=[
            (0, "Utilities bill a business for its highest 15 minutes of power each month."),
            (4.5, "Palo Alto's E-4 tariff: $30.18 + $21.27 for every kW of that peak.")]),
        Scene([Image.open(CARDS / "1_hook.png").convert("RGB").resize((W, H))], 6, zoom=(1.0, 1.03)),
        Scene(meter_frames(10), 10, zoom=(1.0, 1.0), captions=[
            (0, "A real heat-wave day at a real building. Standard smart charging can't see the building..."),
            (5, "...so it stacks its load right on top of the building's evening peak, when every kW costs the most.")]),
        Scene([card([("So owners install fewer chargers,", SEMI, 60, INK),
                      ("or wait for a grid upgrade.", SEMI, 60, INK)])], 4.5, zoom=(1.0, 1.0)),
        Scene([card([("HEADROOM", BOLD, 132, INK), ("An AI operator that fits more chargers", SEMI, 54, GREEN),
                      ("on the grid connection a building already has.", SEMI, 54, GREEN),
                      ("See the building  ·  Plan the charging  ·  Guarantee the limit  ·  Muse Spark runs it",
                       FONT, 34, MUTED)])], 6.5, zoom=(1.0, 1.03)),
        Scene(replay, 21, zoom=(1.0, 1.0), captions=[
            (0, "The same day, 24 chargers, real drivers. Headroom (green) watches the building..."),
            (6.5, "...and charges in the room the building leaves, under the peak the month already paid for."),
            (13, f"Same cars, same energy delivered: {saving:.0%} lower demand charges over the test months.")]),
        Scene([app_shot("spark_vans")], 9, zoom=(1.12, 1.2), focus=(0.5, 0.55), captions=[
            (0, "The site manager just asks, in plain language."),
            (4, "Meta's Muse Spark answers through checked tools: it can't touch the chargers directly.")]),
        Scene([app_shot("spark_expansion")], 10, zoom=(1.12, 1.2), focus=(0.5, 0.62), captions=[
            (0, "\"Can I add 12 more chargers? What would it cost me?\" Answered from simulation on real data."),
            (4.5, "Every number in the answer is traced to a tool result, or the answer is flagged.")]),
        Scene([app_shot("glimmer_status", crop_sidebar=False)], 9, zoom=(1.0, 1.1), focus=(0.4, 0.35), captions=[
            (0, "Internet down? Meta's open-weight Muse Glimmer takes over on the site's own GPU."),
            (4, "Charging never depends on either model: a safety layer guarantees the grid limit.")]),
        Scene([app_shot("check_buildings")], 8, focus=(0.5, 0.35), captions=[
            (0, f"Not just one building: tested on {len(p)} real metered buildings at {p['site_id'].nunique()} US sites."),
            (4, f"More chargers on the same connection in {more:.0%} of them, fewer in none.")]),
        Scene([app_shot("check_sample")], 6, focus=(0.5, 0.75),
              caption="Any owner can upload a meter export and get their own answer."),
        Scene([app_shot("tariff_pge")], 6, focus=(0.4, 0.45),
              caption="Muse Spark reads the utility's tariff, here PG&E's, and it's used only if it reproduces the real bill."),
        Scene([app_shot("proof_ocpp")], 6, focus=(0.4, 0.35),
              caption="Decisions reach chargers over OCPP, the protocol real chargers speak: 480 of 480 accepted."),
        Scene([Image.open(CARDS / "2_result.png").convert("RGB").resize((W, H))], 7, zoom=(1.0, 1.03)),
        Scene([Image.open(CARDS / "3_close.png").convert("RGB").resize((W, H))], 6, zoom=(1.0, 1.02)),
    ]


def main() -> None:
    from make_cards import hero

    hero("hero_1080", (19.2, 10.8), "PROJECT DEMO")
    story = scenes()
    total = sum(s.seconds for s in story) - FADE * (len(story) - 1)
    print(f"{len(story)} scenes, {total:.1f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "headroom_demo.mp4"
    ffmpeg = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                               "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                              stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fade = int(FADE * FPS)
    tail: list[np.ndarray] = []
    for k, scene in enumerate(story):
        frames = (np.asarray(f) for f in render(scene))
        if tail:  # crossfade the previous scene's last frames into this one's first
            for j, prev in enumerate(tail):
                a = (j + 1) / (fade + 1)
                ffmpeg.stdin.write((prev * (1 - a) + next(frames) * a).astype(np.uint8).tobytes())
        buffer: list[np.ndarray] = []
        for f in frames:
            buffer.append(f)
            if len(buffer) > fade:
                ffmpeg.stdin.write(buffer.pop(0).tobytes())
        tail = buffer if k < len(story) - 1 else []
        for f in buffer if k == len(story) - 1 else []:
            ffmpeg.stdin.write(f.tobytes())
        print(f"scene {k + 1}/{len(story)} done", flush=True)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
