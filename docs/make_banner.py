#!/usr/bin/env python3
"""Draw the repo banner. Pure PIL, no generated imagery — house style.

Concept: two engine profiles (native 262K recommended default vs 1M YaRN
on-demand) on two Dell Pro Max with GB10. Left = wordmark. Right = two stacked
slabs for the two context tiers, 01/02 annotations, a dashed down-flow for the
on-demand switch, and a single orange arrow marking the native 262K recommended
default tier (pending owner sign-off).

Requires Pillow (pip install Pillow). Fonts used are the macOS system fonts at
/System/Library/Fonts/HelveticaNeue.ttc and /System/Library/Fonts/Menlo.ttc; if
those paths are missing or the exact face index is unavailable, the script falls
back to ImageFont.load_default(), which renders at a different size and the
layout will not match the committed banner. To reproduce the layout exactly on a
system without those fonts, install the same two font files or substitute fonts
with matching metrics.
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640
BG = (10, 10, 10)
WHITE = (245, 245, 245)
GREY = (140, 140, 140)
DIM = (70, 70, 70)
ORANGE = (255, 122, 26)
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs/assets/banner.png"

HN = "/System/Library/Fonts/HelveticaNeue.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"

def font(path, size, index=0):
    try:
        return ImageFont.truetype(path, size, index=index)
    except Exception:
        return ImageFont.load_default()

f_title = font(HN, 88, 7)     # Light
f_tag = font(HN, 30, 7)
f_mono = font(MENLO, 17)
f_label = font(MENLO, 15)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)

# crosshair corners
for cx, cy in ((40, 40), (W - 40, H - 40)):
    d.line([(cx - 11, cy), (cx + 11, cy)], fill=DIM, width=1)
    d.line([(cx, cy - 11), (cx, cy + 11)], fill=DIM, width=1)

# 5x3 dot lattices
for ox, oy in ((72, 78), (1150, 536)):
    for r in range(3):
        for c in range(5):
            x, y = ox + c * 15, oy + r * 12
            d.ellipse([x, y, x + 1.6, y + 1.6], fill=DIM)

# wordmark
d.text((80, 196), "NATIVE 262K", font=f_title, fill=WHITE)
d.text((80, 288), "OR YARN 1M", font=f_title, fill=WHITE)
d.text((82, 414), "Engine-form A/B on two Dell Pro Max with GB10.", font=f_tag, fill=WHITE)
d.text((82, 462), "QWEN3.8-FLASH-NEXT · VLLM · MTP4 · FP8 KV · MEASURED DEFAULT", font=f_mono, fill=GREY)

# right: two tiers as slabs (isometric-ish parallelograms), stacked
SX, SY = 860, 180
tiers = [("NATIVE 262K", "01"), ("YARN x4  1M", "02")]
slab_w, slab_h, skew, gap = 260, 58, 42, 64
boxes = []
for i, (name, num) in enumerate(tiers):
    y = SY + i * (slab_h + gap)
    poly = [(SX + skew, y), (SX + skew + slab_w, y), (SX + slab_w, y + slab_h), (SX, y + slab_h)]
    d.polygon(poly, outline=(96, 96, 96), width=1)
    d.text((SX + skew + 14, y + 18), name, font=f_label, fill=GREY)
    d.text((SX + skew + slab_w + 16, y + 20), num, font=f_label, fill=DIM)
    boxes.append(y)

# dashed down-flow (on-demand switch) on the left edge of the stack
def dashed(p0, p1, dash=6, gap_=6, fill=DIM):
    x0, y0 = p0; x1, y1 = p1
    L = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    n = int(L // (dash + gap_))
    for k in range(n):
        t0 = k * (dash + gap_) / L; t1 = (k * (dash + gap_) + dash) / L
        d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0), (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=fill, width=1)

lx = SX + 20
dashed((lx, boxes[0] + slab_h), (lx, boxes[1]))
d.text((lx - 70, boxes[0] + slab_h + 18), "on-demand", font=f_label, fill=DIM)

# the single orange marker: native 262K is the shipped default tier
ry = SX + skew + slab_w - 30
ty = boxes[0] + slab_h // 2
d.ellipse([ry - 5, ty - 5, ry + 5, ty + 5], fill=ORANGE)
d.text((ry + 14, ty - 8), "DEFAULT", font=f_label, fill=ORANGE)

# footer rule + labels
d.line([(80, 560), (W - 80, 560)], fill=(40, 40, 40), width=1)
d.text((80, 576), "RYANAI LAB", font=f_label, fill=GREY)
d.text((W - 80 - 250, 576), "DELL PRO MAX WITH GB10", font=f_label, fill=GREY)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print("wrote", OUT)
