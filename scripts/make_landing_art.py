"""Render the landing page's HD artwork (network constellations) into app/landing/img/. Run: python scripts/make_landing_art.py

Everything is drawn from the project's own idea: accounts are nodes, payments are edges, the laundering chain is coral.
Pure numpy + Pillow; seeds are fixed so the images are reproducible.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "app" / "landing" / "img"
BG = (6, 6, 10)
CYAN, MINT, LEMON, CORAL, ORANGE, BLUE = (147, 241, 247), (154, 225, 135), (243, 234, 122), (255, 92, 138), (251, 182, 92), (185, 200, 255)
SS = 2                                                      # supersampling for smooth edges


class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h = w * SS, h * SS
        self.base = Image.new("RGB", (self.w, self.h), BG)
        self.glow = Image.new("RGB", (self.w, self.h), (0, 0, 0))
        self.d, self.g = ImageDraw.Draw(self.base), ImageDraw.Draw(self.glow)

    def line(self, a, b, color, width=2.0, glow=True, alpha=1.0):
        c = tuple(int(v * alpha) for v in color)
        self.d.line([a[0] * SS, a[1] * SS, b[0] * SS, b[1] * SS], fill=c, width=max(1, int(width * SS)))
        if glow:
            self.g.line([a[0] * SS, a[1] * SS, b[0] * SS, b[1] * SS], fill=c, width=max(1, int(width * SS * 3)))

    def dot(self, p, r, color, ring=False, glow=True):
        x, y, rr = p[0] * SS, p[1] * SS, r * SS
        box = [x - rr, y - rr, x + rr, y + rr]
        if ring:
            self.d.ellipse(box, outline=color, width=max(2, int(rr * 0.22)))
        else:
            self.d.ellipse(box, fill=color)
        if glow:
            gr = rr * 2.2
            self.g.ellipse([x - gr, y - gr, x + gr, y + gr], fill=tuple(int(v * 0.55) for v in color))

    def arc(self, center, radius, color, width=1.5, start=0, end=360):
        x, y, r = center[0] * SS, center[1] * SS, radius * SS
        self.d.arc([x - r, y - r, x + r, y + r], start, end, fill=color, width=max(1, int(width * SS)))

    def finish(self, path: Path, size: tuple[int, int], bloom=(10, 38), grain=7, seed=1):
        blur_a = self.glow.filter(ImageFilter.GaussianBlur(bloom[0] * SS))
        blur_b = self.glow.filter(ImageFilter.GaussianBlur(bloom[1] * SS))
        img = ImageChops.screen(self.base, ImageChops.add(blur_a, blur_b, 1.6))
        img = img.resize(size, Image.LANCZOS)
        rng = np.random.default_rng(seed)
        arr = np.asarray(img).astype(np.float32) + rng.normal(0, grain, (size[1], size[0], 1))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=88, optimize=True, progressive=True)
        print("wrote", path.name, path.stat().st_size // 1024, "KB")


def hero(path: Path, size=(2400, 1600)):
    """An account at the centre, rings of counterparties, one coral chain running through them."""
    w, h = size
    c = Canvas(w, h)
    rng = np.random.default_rng(818)
    cx, cy = w * 0.5, h * 0.5
    nodes = []
    for ring, (r, n) in enumerate([(210, 7), (430, 16), (680, 28), (950, 40)]):
        off = rng.uniform(0, 6.28)
        for i in range(n):
            a = off + i * 6.2832 / n + rng.normal(0, 0.05)
            rr = r + rng.normal(0, 22)
            nodes.append((ring, cx + rr * np.cos(a), cy + rr * 0.78 * np.sin(a)))
    for r in (210, 430, 680, 950):
        c.arc((cx, cy), r, (28, 30, 40), 1.4)               # faint orbit; the ellipse squash is applied through points only
    # grey payments between neighbouring rings
    for ring, x, y in nodes:
        near = [n for n in nodes if n[0] in (ring - 1, ring) and (n[1], n[2]) != (x, y)]
        near.sort(key=lambda n: (n[1] - x) ** 2 + (n[2] - y) ** 2)
        for n in near[:2]:
            c.line((x, y), (n[1], n[2]), (46, 50, 66), 1.4, glow=False)
        if ring == 0:
            c.line((cx, cy), (x, y), (70, 80, 100), 1.8, glow=False)
    # the laundering chain
    chain = [(cx, cy)]
    for ring in range(4):
        pool = [n for n in nodes if n[0] == ring]
        pool.sort(key=lambda n: -(n[1] - chain[-1][0]) * 0.9 + abs(n[2] - chain[-1][1]) * 0.3)
        chain.append((pool[0][1], pool[0][2]))
    for a, b in pairwise(chain):
        c.line(a, b, CORAL, 4.2)
    for ring, x, y in nodes:
        flagged = any(abs(x - p[0]) < 1 and abs(y - p[1]) < 1 for p in chain)
        col = CORAL if flagged else [CYAN, CYAN, BLUE, MINT][ring]
        c.dot((x, y), 11 if flagged else 5 + (3 - ring), col, glow=flagged or ring < 2)
    c.dot((cx, cy), 34, CYAN, ring=True)
    c.dot((cx, cy), 14, CYAN)
    c.finish(path, size)


def cycle(path: Path, size=(1600, 1600)):
    """A money loop: accounts on a ring passing the same funds round."""
    w, h = size
    c = Canvas(w, h)
    cx, cy, R, n = w / 2, h / 2, 540, 9
    pts = [(cx + R * np.cos(-1.5708 + i * 6.2832 / n), cy + R * np.sin(-1.5708 + i * 6.2832 / n)) for i in range(n)]
    c.arc((cx, cy), R + 120, (24, 26, 36), 1.4)
    c.arc((cx, cy), R - 160, (24, 26, 36), 1.4)
    for i, p in enumerate(pts):
        q = pts[(i + 1) % n]
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        c.line(p, q, ORANGE, 5)
        v = (q[0] - p[0], q[1] - p[1])
        L = np.hypot(*v)
        ux, uy = v[0] / L, v[1] / L
        tip = (mid[0] + ux * 26, mid[1] + uy * 26)
        c.line((mid[0] - ux * 20 - uy * 22, mid[1] - uy * 20 + ux * 22), tip, ORANGE, 5, glow=False)
        c.line((mid[0] - ux * 20 + uy * 22, mid[1] - uy * 20 - ux * 22), tip, ORANGE, 5, glow=False)
    rng = np.random.default_rng(3)
    for p in pts:                                               # innocent bystanders hanging off the ring
        for _ in range(3):
            a = rng.uniform(0, 6.28)
            q = (p[0] + np.cos(a) * rng.uniform(90, 200), p[1] + np.sin(a) * rng.uniform(90, 200))
            c.line(p, q, (40, 44, 58), 1.6, glow=False)
            c.dot(q, 5, (120, 128, 150), glow=False)
    for p in pts:
        c.dot(p, 26, ORANGE, ring=True)
        c.dot(p, 11, ORANGE)
    c.finish(path, size)


def fan(path: Path, size=(1600, 1600)):
    """Fan-out then gather: one account scatters funds, many small hops, one account collects them."""
    w, h = size
    c = Canvas(w, h)
    rng = np.random.default_rng(11)
    src, dst = (190, h / 2), (w - 190, h / 2)
    mids = [(w / 2 + rng.normal(0, 120), 170 + i * (h - 340) / 13) for i in range(14)]
    for m in mids:
        c.line(src, m, MINT, 2.6)
        c.line(m, dst, CYAN, 2.6)
    for m in mids:
        c.dot(m, 11, LEMON)
    for _ in range(40):                                         # background traffic
        a, b = (rng.uniform(80, w - 80), rng.uniform(80, h - 80)), (rng.uniform(80, w - 80), rng.uniform(80, h - 80))
        c.line(a, b, (30, 33, 44), 1.2, glow=False)
        c.dot(a, 4, (90, 96, 120), glow=False)
    c.dot(src, 44, MINT, ring=True)
    c.dot(src, 18, MINT)
    c.dot(dst, 44, CYAN, ring=True)
    c.dot(dst, 18, CYAN)
    c.finish(path, size)


def ribbon(path: Path, size=(2400, 900)):
    """The 'paper trail': a strip of payments as a timeline, laundering bursts in coral."""
    w, h = size
    c = Canvas(w, h)
    rng = np.random.default_rng(5)
    base = h * 0.62
    for x in range(60, w - 60, 14):
        hot = (700 < x < 980) or (1500 < x < 1660)
        hgt = rng.uniform(40, 360) if hot else rng.uniform(6, 120) * (1 + 1.4 * (rng.random() < 0.06))
        col = CORAL if hot and rng.random() < 0.7 else ((86, 94, 120) if rng.random() < 0.8 else CYAN)
        c.line((x, base), (x, base - hgt), col, 6.5, glow=col == CORAL)
        c.line((x, base + 14), (x, base + 14 + hgt * 0.35), tuple(int(v * 0.35) for v in col), 6.5, glow=False)
    c.line((40, base + 4), (w - 40, base + 4), (60, 66, 86), 2, glow=False)
    c.finish(path, size, grain=5)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hero(OUT / "hero-network.jpg")
    cycle(OUT / "pattern-cycle.jpg")
    fan(OUT / "pattern-scatter-gather.jpg")
    ribbon(OUT / "paper-trail.jpg")
