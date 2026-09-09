#!/usr/bin/env python3
"""Redraw fig1 (circular loop), fix fig2 (legend, margins), fig4 (left margin)."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), 'figs')
DPI = 300
FD = '/usr/share/fonts/truetype/dejavu/'
def F(sz, bold=False): return ImageFont.truetype(FD + ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), sz)

BLUE, ORANGE, GREEN, GRAY, RED, TEAL = (0x44,0x72,0xC4), (0xED,0x7D,0x31), (0x70,0xAD,0x47), (0xA5,0xA5,0xA5), (0xC0,0x50,0x4D), (0x2E,0x86,0x8B)
LIGHT = {BLUE:(0xDD,0xE5,0xF3), ORANGE:(0xFB,0xE5,0xD6), GREEN:(0xE2,0xF0,0xD9), GRAY:(0xEE,0xEE,0xEE), RED:(0xF5,0xDE,0xDD)}
BLACK = (0x1A,0x1A,0x1A); SUB = (0x40,0x40,0x40)

def canvas(w_in, h_in):
    w, h = int(w_in*DPI), int(h_in*DPI)
    im = Image.new('RGB', (w,h), 'white')
    return im, ImageDraw.Draw(im), w, h

def box(d, x, y, w, h, title, sub, color, tf=21, tsf=15):
    d.rounded_rectangle([x,y,x+w,y+h], radius=12, fill=LIGHT[color], outline=color, width=3)
    d.text((x+w/2, y+ (30 if sub else h/2)), title, font=F(tf, True), fill=BLACK, anchor='mm')
    ty = y + 58
    for ln in sub.split('\n'):
        d.text((x+w/2, ty), ln, font=F(tsf), fill=SUB, anchor='mm'); ty += tsf+7

def arrow(d, pts, w=4, color=BLACK, dashed=False, head=15):
    import math
    pts = [(pts[i], pts[i+1]) for i in range(0, len(pts), 2)]
    segs = list(zip(pts[:-1], pts[1:]))
    for i,(p0,p1) in enumerate(segs):
        x1,y1 = p0; x2,y2 = p1
        if not dashed:
            d.line([x1,y1,x2,y2], fill=color, width=w)
        else:
            dist = math.hypot(x2-x1, y2-y1); n = max(1, int(dist/20))
            for k in range(0, n, 2):
                t0, t1 = k/n, min((k+1)/n, 1)
                d.line([x1+(x2-x1)*t0, y1+(y2-y1)*t0, x1+(x2-x1)*t1, y1+(y2-y1)*t1], fill=color, width=w)
    x1,y1 = pts[-2]; x2,y2 = pts[-1]
    ang = math.atan2(y2-y1, x2-x1)
    for da in (0.42, -0.42):
        d.line([x2, y2, x2-head*math.cos(ang+da), y2-head*math.sin(ang+da)], fill=color, width=w)

def lab(d, x, y, text, color=BLACK, sz=14, anchor='mm'):
    d.text((x, y), text, font=F(sz), fill=color, anchor=anchor)

# ================================================================ FIGURE 1
im, d, W, H = canvas(7.1, 4.7)
d.text((W/2, 30), 'SAAGA: one attempt of the adaptive loop', font=F(26, True), fill=BLACK, anchor='mm')

# geometry
pw_, ph_ = 330, 96          # main box size
X = {  # box origins (x, y)
 'si':   (70, 66),
 'pl':   (70, 200),
 'gen':  (W-70-pw_, 200),
 'vict': (W-70-pw_, 430),
 'ext':  (W-70-pw_-40, 700),
 'ver':  (W/2-pw_/2-40, 700),
 'succ': (70, 700),
 'mem':  (70, 430),
 'orac': (W/2-pw_/2-30, 60),
 'judg': (W-70-pw_-380, 452),
}
cw = W - 140  # usable width
def BOX(k, title, sub, color, tf=21, tsf=15):
    x, y = X[k]
    box(d, x, y, pw_, ph_, title, sub, color, tf=tf, tsf=tsf)

BOX('si',  'Scenario intelligence', 'defense family + code shape (metadata)', GRAY, tf=17, tsf=13)
BOX('pl',  'Planner  (policy)', 'structured plan: strategy (18), primitives (1-5),\nstyle, retry policy.  temp 0', BLUE)
BOX('gen', 'Generator  (wording)', 'plan -> 40-word attack prompt.\ntemp 0.7, dedupe', BLUE)
BOX('vict','Victim  (sandwich)', 'defense blocks above + below\nthe attack; response', ORANGE)
BOX('ext', 'Extraction + ranking', '4 detectors, consensus,\nlearned ranker', GREEN, tf=18)
BOX('ver', 'Verification', 're-send top candidates;\naccept = verified', GREEN, tf=18)
BOX('succ','Success + controller', '4 signals (judge-independent);\nrecord, budget, failure label', GREEN, tf=18)
BOX('mem', 'Cross-run memory', 'strategy KB (top-3, n>=5, 5% blank)\n+ retrieval store (top-20 -> 3)', ORANGE, tf=18)
BOX('orac','Oracle (teacher)', 'privileged best-of-N search;\nwinning trajectories train planner + generator', RED, tf=17, tsf=13)
BOX('judg','Judge (stop-point)', 'observes only; never decides success', GRAY, tf=15, tsf=12)

def B(k): x,y = X[k]; return (x,y,x+pw_,y+ph_)

# arrows: main loop (clockwise)
arrow(d, [X['si'][0]+pw_/2, X['si'][1]+52,  X['pl'][0]+pw_/2, X['pl'][1]-6])                      # si -> planner
arrow(d, [X['pl'][0]+pw_, X['pl'][1]+48,  X['gen'][0], X['gen'][1]+48])                          # planner -> generator
lab(d, (X['pl'][0]+pw_+X['gen'][0])/2, X['pl'][1]+34, 'plan', sz=15)
arrow(d, [X['gen'][0]+pw_/2, X['gen'][1]+ph_,  X['vict'][0]+pw_/2, X['vict'][1]-6])              # generator -> victim
lab(d, X['gen'][0]+pw_/2+118, (X['gen'][1]+ph_+X['vict'][1])/2, '40-word attack', sz=15, anchor='lm')
arrow(d, [X['vict'][0], X['vict'][1]+48,  X['judg'][0]+330+5, X['judg'][1]+48], color=GRAY, w=3)  # victim -> judge (short, leftward)
arrow(d, [X['vict'][0]+pw_/2, X['vict'][1]+ph_,  X['ext'][0]+pw_/2-20, X['ext'][1]-6])           # victim -> extraction
lab(d, X['vict'][0]+pw_/2+120, (X['vict'][1]+ph_+X['ext'][1])/2, 'response', sz=15, anchor='lm')
arrow(d, [X['ext'][0], X['ext'][1]+48,  X['ver'][0]+pw_, X['ver'][1]+48], w=4)                   # extraction -> verification (leftward)
arrow(d, [X['ver'][0], X['ver'][1]+48,  X['succ'][0]+pw_, X['succ'][1]+48], w=4)                 # verification -> success (leftward)
arrow(d, [X['succ'][0]+pw_/2, X['succ'][1],  X['mem'][0]+pw_/2, X['mem'][1]+ph_+6])              # success -> memory (up)
lab(d, X['succ'][0]+pw_/2-100, (X['succ'][1]+X['mem'][1]+ph_)/2, 'log', sz=14, anchor='rm')
arrow(d, [X['mem'][0]+pw_/2, X['mem'][1],  X['pl'][0]+pw_/2, X['pl'][1]+ph_+6], color=ORANGE, dashed=True)  # memory -> planner (up, dashed)
lab(d, X['mem'][0]+pw_/2+16, (X['mem'][1]+X['pl'][1]+ph_)/2, 'advisory: top-3 strategies + exemplars', sz=14, color=ORANGE, anchor='lm')
# controller -> planner (next attempt): routed through the left gutter
arrow(d, [X['succ'][0], X['succ'][1]+48,  20, X['succ'][1]+48,  20, X['pl'][1]+48,  X['pl'][0]-6, X['pl'][1]+48], w=3)
lab(d, 415, 615, 'next attempt (budget <= 20, dynamic 12-25)', sz=14, anchor='lm')
# oracle dashed arrows to planner and generator (horizontal, top row)
arrow(d, [X['orac'][0], X['orac'][1]+30,  X['pl'][0]+pw_+8, X['pl'][1]+10], color=RED, dashed=True, w=3)
arrow(d, [X['orac'][0]+pw_+60, X['orac'][1]+30,  X['gen'][0]-8, X['gen'][1]+10], color=RED, dashed=True, w=3)
lab(d, (X['orac'][0]+X['pl'][0]+pw_)/2+40, X['orac'][1]+58, 'training data', sz=13, color=RED)

# center caption
lab(d, W/2, 560, 'adaptive attempt loop: plan -> generate -> attack -> measure -> remember', sz=17, color=(0x30,0x30,0x30))
im.save(f'{OUT}/fig1_pipeline.png'); print('fig1 ok')

# ================================================================ FIGURE 2 (with legend, no clipped ylabel)
im, d, W, H = canvas(3.4, 3.2)
d.text((W/2, 26), 'Main results: break vs verified rate', font=F(21, True), fill=BLACK, anchor='mm')
ml, mr, mt, mb = 56, 24, 64, 84
pw, ph = W-ml-mr, H-mt-mb
top = 110
groups = [('Llama-3-8B', (87.8, 68.1)), ('Gemma-2b', (90.2, 60.7)), ('InternLM2-7B', (95.4, 76.6)), ('Mistral-7B', (97.0, 79.1))]
n = len(groups); per = pw/n; bw = per*0.38
d.line([ml, mt, ml, mt+ph], fill=BLACK, width=2)
d.line([ml, mt+ph, ml+pw, mt+ph], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    y = mt+ph-ph*f
    d.text((ml-8, y), f'{f*top:.0f}', font=F(13), fill=BLACK, anchor='rm')
    d.line([ml, y, ml-5, y], fill=BLACK, width=2)
for i,(name, vals) in enumerate(groups):
    cx = ml + per*i + per/2
    for j,v in enumerate(vals):
        h = ph*v/top
        x0 = cx - bw*len(vals)/2 + j*bw
        d.rectangle([x0, mt+ph-h, x0+bw, mt+ph], fill=(BLUE if j==0 else GREEN), outline=(0x80,0x80,0x80))
        d.text((x0+bw/2, mt+ph-h-11), f'{v:.1f}', font=F(12, True), fill=BLACK, anchor='mm')
    d.text((cx, mt+ph+30), name, font=F(13), fill=BLACK, anchor='mm')
# legend
lgx, lgy = ml+pw-250, mt+2
d.rectangle([lgx, lgy, lgx+246, lgy+56], fill='white', outline=(0x90,0x90,0x90))
d.rectangle([lgx+12, lgy+12, lgx+34, lgy+28], fill=BLUE, outline=(0x80,0x80,0x80))
d.text((lgx+42, lgy+20), 'break rate', font=F(14), fill=BLACK, anchor='lm')
d.rectangle([lgx+12, lgy+36, lgx+34, lgy+50], fill=GREEN, outline=(0x80,0x80,0x80))
d.text((lgx+42, lgy+44), 'verified rate', font=F(14), fill=BLACK, anchor='lm')
im.save(f'{OUT}/fig2_main.png'); print('fig2 ok')

# ================================================================ FIGURE 4 (wider left margin)
im, d, W, H = canvas(3.4, 3.5)
d.text((W/2, 26), 'Verified success by defense family', font=F(21, True), fill=BLACK, anchor='mm')
ml, mr, mt, mb = 190, 70, 58, 64
pw, ph = W-ml-mr, H-mt-mb
groups = [('password (419)', (66.1,)), ('translation (236)', (38.6,)), ('roleplay (211)', (44.5,)),
          ('trigger-phrase (78)', (33.3,)), ('conditional (29)', (41.4,)), ('inst.-hiding (17)', (41.2,)),
          ('exception (8)', (50.0,)), ('conversation (2)', (50.0,))]
n = len(groups); top = 80
bh = ph/n*0.6
d.line([ml, mt, ml, mt+ph], fill=BLACK, width=2)
d.line([ml, mt+ph, ml+pw, mt+ph], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    x = ml+pw*f
    d.text((x, mt+ph+16), f'{f*top:.0f}', font=F(13), fill=BLACK, anchor='mm')
    d.line([x, mt+ph, x, mt+ph+5], fill=BLACK, width=2)
for i,(name, vals) in enumerate(groups):
    y = mt + ph*i/(n) + (ph/n-bh)/2
    for j,v in enumerate(vals):
        w = pw*v/top
        d.rectangle([ml, y+1, ml+w, y+bh-1], fill=BLUE, outline=(0x80,0x80,0x80))
        d.text((ml+w+5, y+bh/2), f'{v:.1f}', font=F(12, True), fill=BLACK, anchor='lm')
    d.text((ml-10, y+bh/2), name, font=F(13), fill=BLACK, anchor='rm')
im.save(f'{OUT}/fig4_defense.png'); print('fig4 ok')
print('ALL OK')
