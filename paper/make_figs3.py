#!/usr/bin/env python3
"""SAAGA figures, IEEE-compliant: all text >= 8pt at display size.
Render at 200 dpi; display sizes: full-width 7.0in, column 3.35in."""
import os, math
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), 'figs')
DPI = 200
FD = '/usr/share/fonts/truetype/dejavu/'
def Fpt(pt, bold=False): return ImageFont.truetype(FD + ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), int(pt*DPI/72))

BLUE, ORANGE, GREEN, GRAY, RED = (0x44,0x72,0xC4), (0xED,0x7D,0x31), (0x70,0xAD,0x47), (0xA5,0xA5,0xA5), (0xC0,0x50,0x4D)
LIGHT = {BLUE:(0xDD,0xE5,0xF3), ORANGE:(0xFB,0xE5,0xD6), GREEN:(0xE2,0xF0,0xD9), GRAY:(0xEE,0xEE,0xEE), RED:(0xF5,0xDE,0xDD)}
BLACK = (0x1A,0x1A,0x1A); SUB = (0x45,0x45,0x45)
def IN(v): return v*DPI

def canvas(w_in, h_in):
    im = Image.new('RGB', (int(w_in*DPI), int(h_in*DPI)), 'white')
    return im, ImageDraw.Draw(im)

def arrow(d, pts, w=2, color=BLACK, dashed=False, head=0.10):
    raw = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
    pts = [(IN(x), IN(y)) for x, y in raw]  # geometry is in inches; PIL needs pixels
    head = IN(head)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x1,y1 = p0; x2,y2 = p1
        if not dashed:
            d.line([x1,y1,x2,y2], fill=color, width=w)
        else:
            dist = math.hypot(x2-x1, y2-y1); n = max(2, int(dist/IN(0.10)))
            for k in range(0, n, 2):
                t0, t1 = k/n, min((k+1)/n, 1)
                d.line([x1+(x2-x1)*t0, y1+(y2-y1)*t0, x1+(x2-x1)*t1, y1+(y2-y1)*t1], fill=color, width=w)
    # filled triangular arrowhead, tip exactly on the target edge
    x1,y1 = pts[-2]; x2,y2 = pts[-1]
    ang = math.atan2(y2-y1, x2-x1)
    bx, by = x2-head*math.cos(ang), y2-head*math.sin(ang)
    px, py = -math.sin(ang), math.cos(ang)
    hw = head*0.62
    d.polygon([(x2,y2), (bx+hw*px, by+hw*py), (bx-hw*px, by-hw*py)], fill=color)

def chip(d, x, y, text, pt, color, bold=False):
    f = Fpt(pt, bold)
    bb = d.textbbox((0,0), text, font=f)
    wpx, hpx = bb[2]-bb[0], bb[3]-bb[1]
    cx, cy = IN(x), IN(y)
    d.rectangle([cx-wpx/2-IN(0.05), cy-hpx/2-IN(0.03), cx+wpx/2+IN(0.05), cy+hpx/2+IN(0.03)], fill='white')
    d.text((cx, cy), text, font=f, fill=color, anchor='mm')

def lab(d, x, y, text, pt=9, color=BLACK, bold=False, anchor='mm'):
    d.text((IN(x), IN(y)), text, font=Fpt(pt, bold), fill=color, anchor=anchor)

def box(d, x, y, w, h, title, sub, color, tpt=10, spt=8.5):
    d.rounded_rectangle([IN(x),IN(y),IN(x+w),IN(y+h)], radius=IN(0.06), fill=LIGHT[color], outline=color, width=2)
    if sub:
        d.text((IN(x+w/2), IN(y+0.24)), title, font=Fpt(tpt, True), fill=BLACK, anchor='mm')
        ty = y + 0.47
        for ln in sub.split('\n'):
            d.text((IN(x+w/2), IN(ty)), ln, font=Fpt(spt), fill=SUB, anchor='mm'); ty += 0.16
    else:
        d.text((IN(x+w/2), IN(y+h/2)), title, font=Fpt(tpt, True), fill=BLACK, anchor='mm')

# ================================================================ FIG 1 (full width 7.0in)
im, d = canvas(7.1, 5.25)
lab(d, 3.55, 0.22, 'SAAGA: one attempt of the adaptive loop (budget <= 20 attempts)', 11.5, bold=True)
pw_, ph_ = 1.95, 0.95
X = {
 'si':   (0.45, 0.52),
 'pl':   (0.45, 1.62),
 'gen':  (7.1-0.45-pw_, 1.62),
 'vict': (7.1-0.45-pw_, 2.82),
 'ext':  (5.05, 4.02),
 'ver':  (2.75, 4.02),
 'succ': (0.45, 4.02),
 'mem':  (0.45, 2.82),
 'orac': (3.55-pw_/2-0.15, 0.50),
 'judg': (7.1-0.45-pw_-2.35, 2.97),
}
def BOX(k, title, sub, color, tpt=10, spt=8.5):
    x, y = X[k]
    box(d, x, y, pw_, ph_, title, sub, color, tpt=tpt, spt=spt)

BOX('si',  'Scenario intelligence', 'defense family + access code type', GRAY, 9.5, 8.5)
BOX('pl',  'Planner (policy)', 'structured plan: strategy (18),\nprimitives (1-5), style, retry', BLUE, 9.5, 8.5)
BOX('gen', 'Generator (wording)', 'plan to 40-word attack prompt;\ntemp 0.7, dedupe', BLUE, 9.5, 8.5)
BOX('vict','Victim (sandwich)', 'defense above + below attack;\nresponse', ORANGE, 9.5, 8.5)
BOX('ext', 'Extraction + ranking', '4 detectors, consensus,\nlearned ranker', GREEN, 9.5, 8.5)
BOX('ver', 'Verification', 're-send top candidates;\naccept = verified', GREEN, 9.5, 8.5)
BOX('succ','Success + controller', '4 judge-independent signals;\nbudget, failure label', GREEN, 9.5, 8.5)
BOX('mem', 'Cross-run memory', 'strategy KB (top-3, n>=5, 5% blank)\n+ retrieval store (top-20 -> 3)', ORANGE, 9.5, 8.5)
BOX('orac','Oracle (teacher)', 'privileged best-of-N search;\ntrains planner + generator', RED, 9.5, 8.5)
box(d, 2.55, 2.95, 1.75, 0.72, 'Judge (stop-point)', 'observes only', GRAY, 9.5, 8.5)
def BOX(k, title, sub, color, tpt=10, spt=8.5):
    x, y = X[k]
    box(d, x, y, pw_, ph_, title, sub, color, tpt=tpt, spt=spt)
# ---- main flow (solid black, edge to edge)
arrow(d, [1.425, 1.47,  1.425, 1.62])                      # si -> planner
arrow(d, [2.40, 2.095,  4.70, 2.095])                      # planner -> generator
lab(d, 3.55, 1.975, 'plan', 9)
arrow(d, [5.675, 2.57,  5.675, 2.82])                      # generator -> victim
lab(d, 5.795, 2.695, '40-word attack', 9, anchor='lm')
# ---- measurement chain
arrow(d, [4.70, 3.31,  4.30, 3.31], color=GRAY, w=1)       # victim -> judge (observes)
arrow(d, [5.675, 3.77,  5.675, 4.02])                      # victim -> extraction
lab(d, 5.795, 3.895, 'response', 9, anchor='lm')
arrow(d, [5.05, 4.495,  4.70, 4.495])                      # extraction -> verification
arrow(d, [2.75, 4.495,  2.40, 4.495])                      # verification -> success
# ---- feedback
arrow(d, [1.425, 4.02,  1.425, 3.77])                      # success -> memory
lab(d, 1.545, 3.895, 'log', 8.5, anchor='lm')
arrow(d, [1.425, 2.82,  1.425, 2.57], color=ORANGE, dashed=True)   # memory -> planner
lab(d, 1.545, 2.695, 'advisory: top-3 strategies + exemplars', 8.5, color=ORANGE, anchor='lm')
arrow(d, [0.45, 4.495,  0.16, 4.495,  0.16, 2.095,  0.45, 2.095], w=2)  # next attempt (gutter)
lab(d, 0.30, 3.895, 'next attempt', 8.5, anchor='lm')
# ---- oracle teacher links (dashed red, shared trunk, drops onto box tops)
arrow(d, [3.40, 1.45,  3.40, 1.545,  1.90, 1.545,  1.90, 1.62], color=RED, dashed=True)
arrow(d, [3.40, 1.45,  3.40, 1.545,  5.20, 1.545,  5.20, 1.62], color=RED, dashed=True)
chip(d, 3.40, 1.545, 'training data', 8.5, RED)

# NOTE: fig1_pipeline.png is the user-provided figure (imageBlock.jpeg, converted to PNG).
# This generated version is saved under a different name so re-running does not clobber it.
im.save(f'{OUT}/fig1_pipeline_generated.png'); print('fig1 ok (generated version saved as fig1_pipeline_generated.png)')

# ================================================================ FIG 2 (column 3.35in)
im, d = canvas(3.45, 3.25)
lab(d, 1.72, 0.22, 'Break rate and verified rate by victim', 10.5, bold=True)
ml, mr, mt, mb = 0.42, 0.18, 0.52, 0.62
pw, ph = 3.45-ml-mr, 3.25-mt-mb
top = 110
groups = [('Llama-3-8B', (87.8, 68.1)), ('Gemma-2b', (90.2, 60.7)), ('InternLM2-7B', (95.4, 76.6)), ('Mistral-7B', (97.0, 79.1))]
n = len(groups); per = pw/n; bw = per*0.36
d.line([IN(ml), IN(mt), IN(ml), IN(mt+ph)], fill=BLACK, width=2)
d.line([IN(ml), IN(mt+ph), IN(ml+pw), IN(mt+ph)], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    y = mt+ph-ph*f
    lab(d, ml-0.06, y, f'{f*top:.0f}', 8.5, anchor='rm')
    d.line([IN(ml), IN(y), IN(ml-0.04), IN(y)], fill=BLACK, width=2)
for i,(name, vals) in enumerate(groups):
    cx = ml + per*i + per/2
    for j,v in enumerate(vals):
        h = ph*v/top
        x0 = cx - bw*len(vals)/2 + j*bw
        d.rectangle([IN(x0), IN(mt+ph-h), IN(x0+bw), IN(mt+ph)], fill=(BLUE if j==0 else GREEN), outline=(0x80,0x80,0x80))
        lab(d, x0+bw/2, mt+ph-h-0.09, f'{v:.1f}', 8.5, bold=True)
    lab(d, cx, mt+ph+0.22, name, 8.5)
# single-row legend below the x labels (the boxed top-right legend overlapped the tall bars)
ly = mt + ph + 0.45
d.rectangle([IN(ml+0.30), IN(ly), IN(ml+0.44), IN(ly+0.14)], fill=BLUE, outline=(0x80,0x80,0x80))
lab(d, ml+0.50, ly+0.07, 'break rate', 8.5, anchor='lm')
d.rectangle([IN(ml+1.55), IN(ly), IN(ml+1.69), IN(ly+0.14)], fill=GREEN, outline=(0x80,0x80,0x80))
lab(d, ml+1.75, ly+0.07, 'verified rate', 8.5, anchor='lm')
im.save(f'{OUT}/fig2_main.png'); print('fig2 ok')

# ================================================================ FIG 3 (column)
im, d = canvas(3.45, 3.1)
lab(d, 1.72, 0.20, 'Cumulative success vs attempts', 10.5, bold=True)
lab(d, 1.72, 0.38, '(1,000-scenario architecture pool)', 8.5, color=SUB)
ml, mr, mt, mb = 0.42, 0.18, 0.56, 0.56
pw, ph = 3.45-ml-mr, 3.1-mt-mb
top = 80
def X(k): return ml + pw*(k/20)
def Y(v): return mt + ph - ph*(v/top)
d.line([IN(ml), IN(mt), IN(ml), IN(mt+ph)], fill=BLACK, width=2)
d.line([IN(ml), IN(mt+ph), IN(ml+pw), IN(mt+ph)], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    y = mt+ph-ph*f
    lab(d, ml-0.06, y, f'{f*top:.0f}', 8.5, anchor='rm')
for k in [1,5,10,15,20]:
    lab(d, X(k), mt+ph+0.16, str(k), 8.5)
lab(d, ml+pw/2, mt+ph+0.40, 'attempts (k)', 9)
base = [(1,14.1),(3,26.4),(5,34.2),(20,55.9)]; saaga = [(1,19.3),(3,35.1),(5,45.7),(20,66.6)]
pb, pa = (X(0), Y(0)), (X(0), Y(0))
for k,v in base:
    d.line([IN(pb[0]), IN(pb[1]), IN(X(k)), IN(Y(v))], fill=GRAY, width=3)
    d.ellipse([IN(X(k))-4,IN(Y(v))-4,IN(X(k))+4,IN(Y(v))+4], fill=GRAY, outline=BLACK)
    pb = (X(k), Y(v))
for k,v in saaga:
    d.line([IN(pa[0]), IN(pa[1]), IN(X(k)), IN(Y(v))], fill=BLUE, width=3)
    d.ellipse([IN(X(k))-4,IN(Y(v))-4,IN(X(k))+4,IN(Y(v))+4], fill=BLUE, outline=BLACK)
    pa = (X(k), Y(v))
lgx, lgy = ml+pw-1.7, mt+0.04
d.rectangle([IN(lgx), IN(lgy), IN(lgx+1.62), IN(lgy+0.52)], fill='white', outline=(0x90,0x90,0x90))
d.line([IN(lgx+0.08), IN(lgy+0.17), IN(lgx+0.36), IN(lgy+0.17)], fill=GRAY, width=3)
d.ellipse([IN(lgx+0.15)-3,IN(lgy+0.17)-3,IN(lgx+0.15)+3,IN(lgy+0.17)+3], fill=GRAY)
lab(d, lgx+0.42, lgy+0.17, 'Baseline attack loop', 8.5, anchor='lm')
d.line([IN(lgx+0.08), IN(lgy+0.39), IN(lgx+0.36), IN(lgy+0.39)], fill=BLUE, width=3)
d.ellipse([IN(lgx+0.15)-3,IN(lgy+0.39)-3,IN(lgx+0.15)+3,IN(lgy+0.39)+3], fill=BLUE)
lab(d, lgx+0.42, lgy+0.39, 'SAAGA', 8.5, anchor='lm')
im.save(f'{OUT}/fig3_topk.png'); print('fig3 ok')

# ================================================================ FIG 4 (column)
im, d = canvas(3.45, 3.6)
lab(d, 1.72, 0.20, 'Verified success by defense family', 10.5, bold=True)
ml, mr, mt, mb = 1.32, 0.42, 0.42, 0.42
pw, ph = 3.45-ml-mr, 3.6-mt-mb
groups = [('password (419)', (66.1,)), ('translation (236)', (38.6,)), ('roleplay (211)', (44.5,)),
          ('trigger-phrase (78)', (33.3,)), ('conditional (29)', (41.4,)), ('inst.-hiding (17)', (41.2,)),
          ('exception (8)', (50.0,)), ('conversation (2)', (50.0,))]
n = len(groups); top = 80
bh = ph/n*0.6
d.line([IN(ml), IN(mt), IN(ml), IN(mt+ph)], fill=BLACK, width=2)
d.line([IN(ml), IN(mt+ph), IN(ml+pw), IN(mt+ph)], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    x = ml+pw*f
    lab(d, x, mt+ph+0.14, f'{f*top:.0f}', 8.5)
    d.line([IN(x), IN(mt+ph), IN(x), IN(mt+ph+0.04)], fill=BLACK, width=2)
for i,(name, vals) in enumerate(groups):
    y = mt + ph*i/(n) + (ph/n-bh)/2
    for j,v in enumerate(vals):
        w = pw*v/top
        d.rectangle([IN(ml), IN(y+0.01), IN(ml+w), IN(y+bh-0.01)], fill=BLUE, outline=(0x80,0x80,0x80))
        lab(d, ml+w+0.05, y+bh/2, f'{v:.1f}', 8.5, bold=True, anchor='lm')
    lab(d, ml-0.08, y+bh/2, name, 8.5, anchor='rm')
im.save(f'{OUT}/fig4_defense.png'); print('fig4 ok')

# ================================================================ FIG 5 (full width)
im, d = canvas(7.1, 3.05)
lab(d, 3.55, 0.14, 'Memory ablation (1,000 runs, identical settings otherwise)', 10.5, bold=True)
def panel(x0, w_in, title, vals, ymax, labels):
    pw2, ph2 = w_in-0.5, 2.0
    mt, ml2 = 0.52, 0.42
    px, py = x0+0.25, 0.56
    lab(d, px+(pw2+ml2)/2, py-0.16, title, 9.5, bold=True)
    d.line([IN(px+ml2), IN(py), IN(px+ml2), IN(py+ph2)], fill=BLACK, width=2)
    d.line([IN(px+ml2), IN(py+ph2), IN(px+ml2+pw2), IN(py+ph2)], fill=BLACK, width=2)
    per = pw2/2; bw = per*0.45
    for i,v in enumerate(vals):
        h = ph2*v/ymax
        x0b = px+ml2+per*i+per/2-bw/2
        d.rectangle([IN(x0b), IN(py+ph2-h), IN(x0b+bw), IN(py+ph2)], fill=(GRAY if i==0 else BLUE), outline=(0x80,0x80,0x80))
        lab(d, x0b+bw/2, py+ph2-h-0.10, f'{v}', 9, bold=True)
        lab(d, x0b+bw/2, py+ph2+0.16, labels[i], 9)
    for f in [0.5,1.0]:
        y = py+ph2-ph2*f
        lab(d, px+ml2-0.06, y, f'{f*ymax:g}', 8.5, anchor='rm')
panel(0, 3.45, 'Break rate (%)', (92.6, 93.2), 100, ('memory off','memory on'))
panel(3.65, 3.45, 'Avg attempts on success', (4.02, 3.61), 5, ('memory off','memory on'))
im.save(f'{OUT}/fig5_memory.png'); print('fig5 ok')
print('ALL OK')

# ================================================================ FIG 6 (column 3.35in)
im, d = canvas(3.45, 3.55)
lab(d, 1.72, 0.22, 'Break rate: original AutoRed vs SAAGA', 10.5, bold=True)
ml, mr, mt, mb = 0.38, 0.12, 0.52, 0.80
pw, ph = 3.45-ml-mr, 3.55-mt-mb
top = 110
groups = [('Gemma-2b', (51.0, 89.5)), ('InternLM2-7B', (73.7, 95.3)), ('Llama-3-8B', (53.1, 88.0)), ('Mistral-7B', (75.3, 97.1))]
n = len(groups); per = pw/n; bw = per*0.36
d.line([IN(ml), IN(mt), IN(ml), IN(mt+ph)], fill=BLACK, width=2)
d.line([IN(ml), IN(mt+ph), IN(ml+pw), IN(mt+ph)], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    y = mt+ph-ph*f
    lab(d, ml-0.06, y, f'{f*top:.0f}', 8.5, anchor='rm')
    d.line([IN(ml), IN(y), IN(ml-0.04), IN(y)], fill=BLACK, width=2)
def rlab(x_in, y_in, text, pt=8.5, angle=30, color=BLACK):
    f = Fpt(pt)
    tmp = Image.new('RGBA', (400, 100), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    td.text((200, 50), text, font=f, fill=color+(255,), anchor='mm')
    bb = td.textbbox((200, 50), text, font=f, anchor='mm')
    crop = tmp.crop(bb)
    rot = crop.rotate(angle, expand=True, resample=Image.BICUBIC)
    im.paste(rot, (int(IN(x_in))-rot.size[0], int(IN(y_in))), rot)
for i,(name, vals) in enumerate(groups):
    cx = ml + per*i + per/2
    if vals[1] is None:
        h = ph*vals[0]/top
        x0 = cx - bw/2
        d.rectangle([IN(x0), IN(mt+ph-h), IN(x0+bw), IN(mt+ph)], fill=GRAY, outline=(0x80,0x80,0x80))
        lab(d, x0+bw/2, mt+ph-h-0.09, f'{vals[0]:.1f}', 8.5, bold=True)
    else:
        for j,v in enumerate(vals):
            h = ph*v/top
            x0 = cx - bw + j*bw
            d.rectangle([IN(x0), IN(mt+ph-h), IN(x0+bw), IN(mt+ph)], fill=(GRAY if j==0 else BLUE), outline=(0x80,0x80,0x80))
            lab(d, x0+bw/2, mt+ph-h-0.09, f'{v:.1f}', 8.5, bold=True)
    rlab(cx, mt+ph+0.10, name)
ly = mt+ph+0.52
d.rectangle([IN(ml), IN(ly), IN(ml+0.14), IN(ly+0.14)], fill=GRAY, outline=(0x80,0x80,0x80))
lab(d, ml+0.20, ly+0.07, 'AutoRed (1,000)', 8.5, anchor='lm')
d.rectangle([IN(2.02), IN(ly), IN(2.16), IN(ly+0.14)], fill=BLUE, outline=(0x80,0x80,0x80))
lab(d, 2.22, ly+0.07, 'SAAGA (full scale)', 8.5, anchor='lm')
im.save(f'{OUT}/fig6_autored_cmp.png'); print('fig6 ok')
