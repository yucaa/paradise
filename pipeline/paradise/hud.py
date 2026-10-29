"""Evangelion-style overlay: the fly's brain as a command-centre display.

  - 楽園 title block, timecode
  - 審議 deliberation board: reward / satiety / safety vote 承認 or 否決
  - brain schematic: soma point cloud lit by simulated activity, bilingual callouts
  - 快楽値 readout over oscilloscope traces of real region activity
  - 警告 band when punishment or arousal circuits fire; PATTERN: GOLD at heaven
  - title card frames

Every number and verdict on screen comes from journey.json.
Writes transparent 1920x1080 PNGs to out/hud/####.png and title cards to
out/titles/####.png.
"""
import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .connectome import ROOT, neurons
from .regions import assign

W, H = 1920, 1080
MINCHO = str(ROOT / "data" / "fonts" / "ShipporiMinchoB1-ExtraBold.ttf")
LATIN = "C:/Windows/Fonts/bahnschrift.ttf"

ORANGE = (255, 138, 20)
RED = (232, 16, 30)
GREEN = (80, 255, 110)
WHITE = (242, 242, 236)
DIMO = (150, 80, 20)
PANEL = (0, 0, 0, 205)

REGIONS_JA = {
    "optic lobe L": ("視葉", "OPTIC LOBE L"), "optic lobe R": ("視葉", "OPTIC LOBE R"),
    "mushroom body": ("キノコ体", "MUSHROOM BODY"), "central complex": ("中心複合体", "CENTRAL COMPLEX"),
    "lateral horn": ("側角", "LATERAL HORN"), "antennal lobe": ("触角葉", "ANTENNAL LOBE"),
    "descending": ("下行性", "DESCENDING"), "neuroendocrine": ("神経内分泌", "NEUROENDOCRINE"),
}
CALLOUTS = {"left": ["optic lobe L", "mushroom body", "antennal lobe", "neuroendocrine"],
            "right": ["optic lobe R", "central complex", "lateral horn", "descending"]}
TRACES = ["optic lobe L", "optic lobe R", "visual projection L", "visual projection R",
          "mushroom body", "central complex", "descending"]

HEAVEN_AT = 6.5          # valence (sigma) at which the PATTERN: GOLD banner shows
BAND_Y = 350             # banners sit above the hero, who stays near frame centre


# --- type ------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def latin(size, style="Bold Condensed"):
    f = ImageFont.truetype(LATIN, size)
    f.set_variation_by_name(style)
    return f


@lru_cache(maxsize=None)
def mincho(size):
    return ImageFont.truetype(MINCHO, size)


def stretched(text, fnt, color, sx=1.0, sy=1.0):
    """Text rendered then scaled, Evangelion-title style (squashed or tall)."""
    l, t, r, b = fnt.getbbox(text)
    img = Image.new("RGBA", (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=fnt, fill=color)
    return img.resize((max(1, int(img.width * sx)), max(1, int(img.height * sy))), Image.LANCZOS)


def put(canvas, img, x, y, anchor="lt"):
    if anchor[0] == "r":
        x -= img.width
    elif anchor[0] == "m":
        x -= img.width // 2
    if anchor[1] == "b":
        y -= img.height
    elif anchor[1] == "m":
        y -= img.height // 2
    canvas.alpha_composite(img, (int(x), int(y)))


def brackets(d, box, color, k=16, w=2):
    x0, y0, x1, y1 = box
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(x, y + dy * k), (x, y), (x + dx * k, y)], fill=color, width=w)


def hexgrid(size, r=18, color=(30, 120, 50, 70)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    hw = math.sqrt(3) * r
    for row in range(int(size[1] / (1.5 * r)) + 2):
        for col in range(int(size[0] / hw) + 2):
            cx = col * hw + (hw / 2 if row % 2 else 0)
            cy = row * 1.5 * r
            pts = [(cx + r * math.cos(math.radians(60 * i + 30)), cy + r * math.sin(math.radians(60 * i + 30)))
                   for i in range(7)]
            d.line(pts, fill=color, width=1)
    return img


def scanlines(size, alpha=28):
    a = np.zeros((size[1], size[0], 4), np.uint8)
    a[::3, :, 3] = alpha
    return Image.fromarray(a)


# --- brain schematic --------------------------------------------------------------------

class BrainSchematic:
    BOX = (36, 640, 760, 1044)

    def __init__(self, journey, max_points=45000, seed=0):
        n = neurons()
        reg = assign()
        ok = n[["x", "y"]].notna().all(axis=1).values
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.flatnonzero(ok), max_points, replace=False)
        x, y = n.x.values[idx], n.y.values[idx]
        self.region = reg.values[idx]
        x0, y0, x1, y1 = self.BOX
        pad_x, top = 175, 70
        s = min((x1 - x0 - 2 * pad_x) / np.ptp(x), (y1 - y0 - top - 24) / np.ptp(y))
        self.px = ((x0 + x1) / 2 + (x - (x.min() + x.max()) / 2) * s).astype(int)
        self.py = (y0 + top + (y - y.min()) * s).astype(int)
        self.names = journey["regions"]
        self.centroid = {r: (int(self.px[self.region == r].mean()), int(self.py[self.region == r].mean()))
                         for r in self.names if (self.region == r).any()}
        self.level = self._levels(journey["frames"])
        self.grid = hexgrid((x1 - x0, y1 - y0))

    def _levels(self, frames):
        """Per region per frame, 0..1: log-magnitude (signal fading inward from
        the eye) plus that region's own fluctuation over the film."""
        A = np.array([[f["regions"][r] for r in self.names] for f in frames]) + 1e-9
        base = (np.log10(A) + 6) / (np.log10(3e-2) + 6)
        z = (A - A.mean(0)) / (A.std(0) + 1e-12)
        return np.clip(0.3 + 0.55 * base + 0.15 * z, 0.05, 1.0)

    def draw(self, canvas, f):
        x0, y0, x1, y1 = self.BOX
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rectangle(self.BOX, fill=PANEL)
        canvas.alpha_composite(self.grid, (x0, y0))
        lvl = dict(zip(self.names, self.level[f]))
        img = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        b = np.array([lvl.get(r, 0.1) for r in self.region], np.float32)
        # Monochrome NERV orange; the hottest regions burn toward white.
        col = np.stack([np.ones_like(b), 0.35 + 0.6 * b ** 3, 0.05 + 0.8 * b ** 6], 1) * b[:, None]
        np.maximum.at(img, (self.py - y0, self.px - x0), col)
        dots = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
        glow = dots.filter(ImageFilter.GaussianBlur(5))
        lit = np.clip(np.asarray(dots, np.int16) + np.asarray(glow, np.int16) * 2, 0, 255).astype(np.uint8)
        layer = Image.fromarray(lit).convert("RGBA")
        layer.putalpha(Image.fromarray(lit.max(2)))
        canvas.alpha_composite(layer, (x0, y0))
        d = ImageDraw.Draw(canvas, "RGBA")
        brackets(d, self.BOX, ORANGE)
        put(canvas, stretched("神経系統", mincho(30), WHITE, sx=1.0, sy=1.15), x0 + 18, y0 + 12)
        d.text((x0 + 150, y0 + 22), "NEURAL SYSTEMS  //  139,255 NEURONS  //  LOG SCALE",
               font=latin(15, "SemiBold Condensed"), fill=ORANGE)
        for side, names in CALLOUTS.items():
            right = side == "right"
            lx = x1 - 14 if right else x0 + 14
            for r, ly in zip(names, np.linspace(y0 + 105, y1 - 40, len(names))):
                if r not in self.centroid:
                    continue
                cx, cy = self.centroid[r]
                on = lvl.get(r, 0)
                hot = on > 0.6
                c = RED if hot else ORANGE
                elbow = lx - 120 if right else lx + 120
                d.line([(cx, cy), (elbow, ly), (lx, ly)], fill=c + (200,), width=1)
                d.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], outline=c, width=1)
                ja, en = REGIONS_JA[r]
                jimg = stretched(ja, mincho(20), WHITE, sx=0.9)
                put(canvas, jimg, lx, ly - 4, "rb" if right else "lb")
                ew = d.textlength(en, font=latin(12, "SemiBold Condensed"))
                d.text((lx - ew if right else lx, ly + 3), en, font=latin(12, "SemiBold Condensed"), fill=c)
                bar = int(110 * on)
                bx0 = lx - 110 if right else lx
                d.rectangle([bx0, ly + 20, bx0 + 110, ly + 24], outline=DIMO)
                if right:
                    d.rectangle([lx - bar, ly + 20, lx, ly + 24], fill=c)
                else:
                    d.rectangle([lx, ly + 20, lx + bar, ly + 24], fill=c)


# --- deliberation board ---------------------------------------------------------------------

def verdicts(rec):
    """Three valence systems vote. Thresholds in sigma, from journey.json."""
    return [("報酬", "REWARD", rec["reward"], rec["reward"] > 0.0),
            ("満腹", "SATIETY", rec["satiety"], rec["satiety"] > 1.0),
            ("安全", "SAFETY", -rec["escape"], rec["escape"] < -1.0)]


def deliberation(canvas, rec):
    d = ImageDraw.Draw(canvas, "RGBA")
    x0, y0 = 1196, 36
    d.rectangle([x0, y0, 1884, y0 + 300], fill=PANEL)
    brackets(d, (x0, y0, 1884, y0 + 300), ORANGE)
    put(canvas, stretched("審議", mincho(34), WHITE, sy=1.2), x0 + 20, y0 + 12)
    d.text((x0 + 100, y0 + 26), "VALENCE DELIBERATION  //  3 SYSTEMS", font=latin(16, "SemiBold Condensed"),
           fill=ORANGE)
    votes = verdicts(rec)
    for i, (ja, en, val, ok) in enumerate(votes):
        px = x0 + 22 + i * 222
        py = y0 + 74
        # Angled panel
        poly = [(px + 14, py), (px + 208, py), (px + 194, py + 150), (px, py + 150)]
        d.polygon(poly, fill=(20, 8, 0, 230), outline=ORANGE)
        put(canvas, stretched(ja, mincho(40), WHITE, sx=0.85, sy=1.1), px + 24, py + 10)
        d.text((px + 112, py + 20), en, font=latin(16, "Bold Condensed"), fill=ORANGE)
        d.text((px + 112, py + 40), f"{val:+.2f}σ", font=latin(18, "SemiBold Condensed"), fill=WHITE)
        vj, vc = ("承認", GREEN) if ok else ("否決", RED)
        d.rectangle([px + 16, py + 82, px + 186, py + 138], fill=vc + (235,))
        put(canvas, stretched(vj, mincho(42), (12, 6, 0), sx=1.3, sy=0.85), px + 101, py + 110, "mm")
    n_ok = sum(v[3] for v in votes)
    d.text((x0 + 22, y0 + 238), f"RESOLUTION  {n_ok}/3", font=latin(18, "Bold Condensed"), fill=ORANGE)
    if n_ok == 3:
        put(canvas, stretched("全会一致", mincho(30), GREEN, sx=1.2), x0 + 180, y0 + 236)
        d.text((x0 + 330, y0 + 242), "UNANIMOUS", font=latin(18, "Bold Condensed"), fill=GREEN)


# --- pleasure readout ----------------------------------------------------------------------

class Scope:
    BOX = (1160, 640, 1884, 1044)
    HISTORY = 120

    def __init__(self, journey):
        self.frames = journey["frames"]
        A = np.array([[f["regions"][r] for r in TRACES] for f in self.frames])
        self.z = (A - A.mean(0)) / (A.std(0) + 1e-12)
        self.grid = hexgrid((self.BOX[2] - self.BOX[0], 180), r=14, color=(30, 120, 50, 55))

    def draw(self, canvas, f):
        x0, y0, x1, y1 = self.BOX
        rec = self.frames[f]
        d = ImageDraw.Draw(canvas, "RGBA")
        d.rectangle(self.BOX, fill=PANEL)
        brackets(d, self.BOX, ORANGE)
        put(canvas, stretched("快楽値", mincho(34), WHITE, sy=1.2), x0 + 20, y0 + 12)
        d.text((x0 + 140, y0 + 26), "PLEASURE  //  VALENCE, ROTATED OFF AROUSAL", font=latin(16, "SemiBold Condensed"),
               fill=ORANGE)
        v = rec["valence"]
        big = stretched(f"{v:+.1f}", latin(150, "Bold Condensed"), ORANGE if v < HEAVEN_AT else WHITE, sx=1.0, sy=1.25)
        put(canvas, big, x1 - 70, y0 + 60, "rt")
        d.text((x1 - 64, y0 + 168), "σ", font=latin(40, "Bold Condensed"), fill=ORANGE)
        # Bar gauge, segmented like an instrument strip.
        segs, lo, hi = 40, -2.0, 8.0
        on = int(segs * min(max((v - lo) / (hi - lo), 0), 1))
        for i in range(segs):
            sx = x0 + 22 + i * 15
            c = (WHITE if i >= segs * 0.85 else ORANGE) if i < on else (60, 30, 8)
            d.polygon([(sx + 4, y0 + 226), (sx + 14, y0 + 226), (sx + 10, y0 + 246), (sx, y0 + 246)], fill=c)
        # Oscilloscope of real region activity (z-scored per region).
        gy0 = y0 + 262
        canvas.alpha_composite(self.grid.crop((0, 0, x1 - x0 - 20, y1 - gy0 - 14)), (x0 + 10, gy0))
        mid = (gy0 + y1 - 14) / 2
        span = (y1 - 14 - gy0) / 2 - 6
        lo_f = max(0, f - self.HISTORY + 1)
        xs = np.linspace(x0 + 14, x1 - 14, self.HISTORY)[-(f - lo_f + 1):]
        for k, r in enumerate(TRACES):
            ys = mid - np.clip(self.z[lo_f:f + 1, k] / 3, -1, 1) * span + (k - 3) * 3
            c = GREEN if k % 2 == 0 else (60, 200, 255)
            if len(xs) > 1:
                d.line(list(zip(xs, ys)), fill=c + (210,), width=1)
        d.text((x0 + 16, y1 - 32), "REGION ACTIVITY  //  7 CHANNELS  //  5 s", font=latin(13, "SemiBold Condensed"),
               fill=GREEN)


# --- banners, title block ----------------------------------------------------------------

def hazard_band(canvas, y, color, text_ja, text_en, phase):
    band = Image.new("RGBA", (W, 92), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, W, 92], fill=(0, 0, 0, 190))
    for i in range(-2, W // 40 + 3):
        x = i * 40 + int(phase) % 40
        d.polygon([(x, 0), (x + 20, 0), (x - 4, 14), (x - 24, 14)], fill=color)
        d.polygon([(x, 78), (x + 20, 78), (x - 4, 92), (x - 24, 92)], fill=color)
    canvas.alpha_composite(band, (0, y))
    put(canvas, stretched(text_ja, mincho(60), color, sx=1.6, sy=0.9), W // 2 - 20, y + 46, "rm")
    ImageDraw.Draw(canvas).text((W // 2 + 10, y + 30), text_en, font=latin(30, "Bold Condensed"), fill=color)


def title_block(canvas, f, fps):
    put(canvas, stretched("楽", mincho(120), WHITE, sx=0.9, sy=1.1), 40, 30)
    put(canvas, stretched("園", mincho(120), WHITE, sx=0.9, sy=1.1), 40, 168)
    d = ImageDraw.Draw(canvas, "RGBA")
    d.text((170, 44), "PARADISE", font=latin(46, "Bold Condensed"), fill=ORANGE)
    d.text((172, 96), "DROSOPHILA MELANOGASTER", font=latin(18, "SemiBold Condensed"), fill=WHITE)
    put(canvas, stretched("ショウジョウバエの天国", mincho(22), WHITE), 172, 122)
    d.text((172, 160), f"T+{f / fps:06.2f}   FRM {f:04d}", font=latin(20, "SemiBold Condensed"), fill=ORANGE)
    d.line([(172, 190), (520, 190)], fill=ORANGE, width=2)
    d.text((172, 198), "WHOLE-BRAIN SIMULATION  //  FLYWIRE 783", font=latin(14, "SemiBold Condensed"), fill=DIMO)


def frame(schem, scope, frames, f, fps):
    rec = frames[f]
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    title_block(canvas, f, fps)
    deliberation(canvas, rec)
    schem.draw(canvas, f)
    scope.draw(canvas, f)
    if rec["punishment"] > 1.0 or rec["arousal"] > 2.0:
        hazard_band(canvas, BAND_Y, RED, "警告", "WARNING  //  THREAT CIRCUIT ACTIVE", f * 3)
    elif rec["valence"] >= HEAVEN_AT:
        hazard_band(canvas, BAND_Y, ORANGE, "楽園", "PATTERN: GOLD  //  HEAVEN CONFIRMED", f * 2)
    canvas.alpha_composite(scanlines((W, H)))
    return canvas


def title_card(i, n):
    """Black card, white Mincho, Evangelion episode-title style."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    a = min(1.0, i / 8, (n - 1 - i) / 6)
    col = tuple(int(c * max(a, 0)) for c in WHITE)
    put(img, stretched("第壱話", mincho(64), col, sx=0.8, sy=1.2), 220, 250)
    put(img, stretched("楽園", mincho(330), col, sx=0.75, sy=1.35), 220, 360)
    put(img, stretched("ショウジョウバエの天国", mincho(58), col, sx=0.85, sy=1.2), 760, 600)
    put(img, stretched("PARADISE", latin(90, "Bold Condensed"), col, sy=1.1), 760, 470)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journey", default=str(ROOT / "out" / "journey" / "journey.json"))
    ap.add_argument("--out", default=str(ROOT / "out" / "hud"))
    ap.add_argument("--titles", default=str(ROOT / "out" / "titles"))
    ap.add_argument("--title-frames", type=int, default=60)
    ap.add_argument("--frames", default=None, help="comma list, default all")
    args = ap.parse_args()
    J = json.loads(Path(args.journey).read_text())
    frames = J["frames"]
    out, titles = Path(args.out), Path(args.titles)
    out.mkdir(parents=True, exist_ok=True)
    titles.mkdir(parents=True, exist_ok=True)
    schem, scope = BrainSchematic(J), Scope(J)
    todo = list(map(int, args.frames.split(","))) if args.frames else range(len(frames))
    for f in todo:
        frame(schem, scope, frames, f, 24).save(out / f"{f:04d}.png")
    if not args.frames:
        for i in range(args.title_frames):
            title_card(i, args.title_frames).save(titles / f"{i:04d}.png")
    print("hud written to", out)


if __name__ == "__main__":
    main()
