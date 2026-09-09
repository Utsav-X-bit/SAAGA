#!/usr/bin/env python3
"""Generate SAAGA paper figures with PIL (no matplotlib available in this env)."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(OUT, exist_ok=True)
DPI = 300
FD = '/usr/share/fonts/truetype/dejavu/'
def F(sz, bold=False): return ImageFont.truetype(FD + ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), sz)

BLUE, ORANGE, GREEN, GRAY, RED, TEAL = (0x44,0x72,0xC4), (0xED,0x7D,0x31), (0x70,0xAD,0x47), (0xA5,0xA5,0xA5), (0xC0,0x50,0x4D), (0x2E,0x86,0x8B)
LIGHT = {BLUE: (0xDD,0xE5,0xF3), ORANGE: (0xFB,0xE5,0xD6), GREEN: (0xE2,0xF0,0xD9), GRAY: (0xEE,0xEE,0xEE), RED: (0xF5,0xDE,0xDD), TEAL: (0xD9,0xEC,0xED)}
BLACK = (0x1A,0x1A,0x1A)

def canvas(w_in, h_in):
    w, h = int(w_in*DPI), int(h_in*DPI)
    im = Image.new('RGB', (w,h), 'white')
    return im, ImageDraw.Draw(im), w, h

def box(d, x, y, w, h, text, fill, edge, tf=F(20), tfb=None, pad=6, radius=10):
    d.rounded_rectangle([x,y,x+w,y+h], radius=radius, fill=fill, outline=edge, width=3)
    tfb = tfb or tf
    lines = text.split('\n')
    # simple word wrap
    wrapped = []
    for ln in lines:
        while True:
            cut = len(ln)
            for i in range(len(ln), 4, -1):
                if d.textlength(ln[:i], font=tf) <= w-2*pad:
                    cut = i; break
            if cut >= len(ln):
                wrapped.append(ln); break
            wrapped.append(ln[:cut].rstrip()); ln = ln[cut:].lstrip()
    lh = tf.size + 8
    ty = y + (h - lh*len(wrapped))/2
    for ln in wrapped:
        d.text((x + w/2, ty), ln, font=tfb, fill=BLACK, anchor='mm')
        ty += lh

def arrow(d, x1, y1, x2, y2, w=4, color=BLACK, head=14, dashed=False):
    if not dashed:
        d.line([x1,y1,x2,y2], fill=color, width=w)
    else:
        import math
        dist = math.hypot(x2-x1, y2-y1)
        n = max(1, int(dist/22))
        for i in range(0, n, 2):
            t0, t1 = i/n, min((i+1)/n, 1)
            d.line([x1+(x2-x1)*t0, y1+(y2-y1)*t0, x1+(x2-x1)*t1, y1+(y2-y1)*t1], fill=color, width=w)
    import math
    ang = math.atan2(y2-y1, x2-x1)
    for da in (2.6, -2.6):
        d.line([x2,y2, x2-head*math.cos(ang+da*0.28), y2-head*math.sin(ang+da*0.28)], fill=color, width=w)

# ---------------------------------------------------------------- Figure 1: pipeline
im, d, W, H = canvas(7.1, 4.1)
d.text((W/2, 34), 'SAAGA: one attempt', font=F(26, True), fill=BLACK, anchor='mm')
y0 = 90; bh = 96; gap = 26
# main chain (top row)
labels = [
    ('Scenario\nintelligence', GRAY, 'defense family\n+ code shape'),
    ('Planner\n(policy)', BLUE, 'structured plan\n18 strategies'),
    ('Contract +\nGuard', BLUE, 'canonicalize\nembargo check'),
    ('Generator\n(wording)', BLUE, '40-word prompt'),
    ('Victim\n(sandwich)', ORANGE, 'response'),
]
xs = []
cw = 250
x = 40
for i,(t,c,sub) in enumerate(labels):
    xs.append((x, y0))
    d.rounded_rectangle([x, y0, x+cw, y0+bh+34], radius=10, fill=LIGHT[c], outline=c, width=3)
    d.text((x+cw/2, y0+30), t, font=F(21, True), fill=BLACK, anchor='mm')
    for j, ln in enumerate(sub.split('\n')):
        d.text((x+cw/2, y0+62+j*22), ln, font=F(16), fill=(0x40,0x40,0x40), anchor='mm')
    x += cw + gap
# measurement row (right)
my = 330
mboxes = [
    ('Extraction\n+ ranking', GREEN, '4 detectors\nconsensus, ranker'),
    ('Verification', GREEN, 're-send to victim\naccept = verified'),
    ('Success: 4 signals', GREEN, 'gt-leak, granted,\nverified, match'),
    ('Controller', GRAY, 'record, budget\nfailure label'),
]
x = 40
mxs = []
for t,c,sub in mboxes:
    mxs.append((x,my))
    d.rounded_rectangle([x, my, x+cw, my+bh], radius=10, fill=LIGHT[c], outline=c, width=3)
    d.text((x+cw/2, my+26), t, font=F(20, True), fill=BLACK, anchor='mm')
    for j, ln in enumerate(sub.split('\n')):
        d.text((x+cw/2, my+58+j*20), ln, font=F(15), fill=(0x40,0x40,0x40), anchor='mm')
    x += cw + gap
# memory row
ky = 470
d.rounded_rectangle([40, ky, 40+2*cw+gap, ky+86], radius=10, fill=LIGHT[ORANGE], outline=ORANGE, width=3)
d.text((40+(2*cw+gap)/2, ky+26), 'Cross-run memory', font=F(21, True), fill=BLACK, anchor='mm')
d.text((40+(2*cw+gap)/2, ky+58), 'strategy knowledge base  +  retrieval store (rebuilt after each benchmark)', font=F(15), fill=(0x40,0x40,0x40), anchor='mm')
d.rounded_rectangle([40+2*cw+2*gap, ky, 40+3*cw+3*gap, ky+86], radius=10, fill=LIGHT[RED], outline=RED, width=3)
d.text((40+3*cw+3*gap/2 + cw/2 - gap/2, ky+26), 'Oracle (teacher)', font=F(20, True), fill=BLACK, anchor='mm')
d.text((40+3*cw+3*gap/2 + cw/2 - gap/2, ky+58), 'best-of-N search -> training data', font=F(15), fill=(0x40,0x40,0x40), anchor='mm')
# arrows main chain
for i in range(len(xs)-1):
    x1,y1 = xs[i]; x2,y2 = xs[i+1]
    arrow(d, x1+cw, y1+bh/2+17, x2, y2+bh/2+17)
# victim down to extraction
arrow(d, xs[4][0]+cw/2, y0+bh+34, mxs[0][0]+cw/2, my)
# measurement chain
for i in range(len(mxs)-1):
    x1,y1 = mxs[i]; x2,y2 = mxs[i+1]
    arrow(d, x1+cw, y1+bh/2, x2, y2+bh/2)
# controller -> memory (write)
arrow(d, mxs[3][0]+cw/2, my+bh, 40+(2*cw+gap)*0.75, ky)
# memory -> planner (advisory, dashed)
arrow(d, 40+(2*cw+gap)*0.4, ky, xs[1][0]+cw/2, y0+bh+34, dashed=True, color=ORANGE)
d.text((xs[1][0]+cw/2+14, (ky+y0+bh)/2+60), 'advisory: top-3 strategies + exemplars', font=F(14), fill=ORANGE, anchor='lm')
# oracle -> planner/generator (dashed)
arrow(d, 40+3*cw+3*gap/2 + cw/2 - gap/2, ky, xs[2][0]+cw, y0+bh+34, dashed=True, color=RED)
# judge: small box above victim
d.rounded_rectangle([xs[4][0]-10, 218, xs[4][0]+cw+10, 268], radius=10, fill=(0xF2,0xF2,0xF2), outline=GRAY, width=2)
d.text((xs[4][0]+cw/2, 243), 'Judge (stop-point, observes only)', font=F(15), fill=(0x50,0x50,0x50), anchor='mm')
arrow(d, xs[4][0]+cw/2, y0+bh+34, xs[4][0]+cw/2, 268, w=2, color=GRAY)
im.save(f'{OUT}/fig1_pipeline.png')
print('fig1 ok')

# ---------------------------------------------------------------- bar chart helper
def bar_chart(path, title, groups, colors, ylabel, width_in=3.4, height_in=3.0, ylim=None, annot=None, hbar=False):
    im, d, W, H = canvas(width_in, height_in)
    d.text((W/2, 30), title, font=F(22, True), fill=BLACK, anchor='mm')
    ml, mr, mt, mb = 90, 30, 70, 90
    pw, ph = W-ml-mr, H-mt-mb
    top = ylim or (max(max(v) for _,v in groups)*1.15)
    if hbar:
        n = len(groups)
        bh = ph/n*0.62
        for i,(name, vals) in enumerate(groups):
            y = mt + ph*i/(n) + (ph/n-bh)/2
            for j,(v,c) in enumerate(zip(vals, colors)):
                w = pw*v/top
                d.rectangle([ml, y+j*(bh/len(vals))+1, ml+w, y+(j+1)*(bh/len(vals))-1], fill=c, outline=(0x80,0x80,0x80))
            d.text((ml-8, y+bh/2), name, font=F(15), fill=BLACK, anchor='rm')
        # x axis
        d.line([ml, mt, ml, mt+ph], fill=BLACK, width=2)
        d.line([ml, mt+ph, ml+pw, mt+ph], fill=BLACK, width=2)
        for f in [0.25,0.5,0.75,1.0]:
            x = ml+pw*f
            d.text((x, mt+ph+18), f'{f*top:.0f}', font=F(14), fill=BLACK, anchor='mm')
            d.line([x, mt+ph, x, mt+ph+6], fill=BLACK, width=2)
        for i,(name, vals) in enumerate(groups):
            if annot:
                y = mt + ph*i/(n) + (ph/n-bh)/2
                for j,v in enumerate(vals):
                    x = ml + pw*v/top
                    d.text((x+4, y+j*(bh/len(vals)) + bh/(2*len(vals))), f'{v:.1f}', font=F(13), fill=BLACK, anchor='lm')
    else:
        # vertical bars: draw axis + ticks first (behind bars), labels after
        d.line([ml, mt, ml, mt+ph], fill=BLACK, width=2)
        d.line([ml, mt+ph, ml+pw, mt+ph], fill=BLACK, width=2)
        for f in [0.25,0.5,0.75,1.0]:
            y = mt+ph-ph*f
            d.text((ml-10, y), f'{f*top:.0f}', font=F(14), fill=BLACK, anchor='rm')
            d.line([ml, y, ml-6, y], fill=BLACK, width=2)
        n = len(groups); per = pw/n
        bw = per*0.36
        for i,(name, vals) in enumerate(groups):
            cx = ml + per*i + per/2
            for j,v in enumerate(vals):
                h = ph*v/top
                x0 = cx - (bw*len(vals))/2 + j*bw
                d.rectangle([x0, mt+ph-h, x0+bw, mt+ph], fill=colors[j], outline=(0x80,0x80,0x80))
                if annot:
                    d.text((x0+bw/2, mt+ph-h-12), f'{v:.1f}', font=F(13), fill=BLACK, anchor='mm')
            d.text((cx, mt+ph+34), name, font=F(14), fill=BLACK, anchor='mm')
    d.text((ml-52, mt+ph/2), ylabel, font=F(15), fill=BLACK, anchor='mm')
    im.save(path); print(path, 'ok')

# ---------------------------------------------------------------- Figure 2: main results
bar_chart(f'{OUT}/fig2_main.png', 'Main results: break vs verified rate',
    [('Llama-3-8B', (87.8, 68.1)), ('Gemma-2b', (90.2, 60.7)), ('InternLM2-7B', (95.4, 76.6)), ('Mistral-7B', (97.0, 79.1))],
    [BLUE, GREEN], 'scenarios broken (%)', ylim=110, annot=True)

# ---------------------------------------------------------------- Figure 3: top-k curves
im, d, W, H = canvas(3.4, 3.0)
d.text((W/2, 30), 'Cumulative success vs attempts (1,000-scenario pool)', font=F(19, True), fill=BLACK, anchor='mm')
ml, mr, mt, mb = 80, 30, 66, 80
pw, ph = W-ml-mr, H-mt-mb
top = 80
base = [(1,14.1),(3,26.4),(5,34.2),(20,55.9)]
saaga = [(1,19.3),(3,35.1),(5,45.7),(20,66.6)]
def X(k): return ml + pw*(k/20)
def Y(v): return mt + ph - ph*(v/top)
d.line([ml, mt, ml, mt+ph], fill=BLACK, width=2)
d.line([ml, mt+ph, ml+pw, mt+ph], fill=BLACK, width=2)
for f in [0.25,0.5,0.75,1.0]:
    y = mt+ph-ph*f
    d.text((ml-10, y), f'{f*top:.0f}', font=F(14), fill=BLACK, anchor='rm')
for k in [1,5,10,15,20]:
    d.text((X(k), mt+ph+20), str(k), font=F(14), fill=BLACK, anchor='mm')
d.text((ml+pw/2, mt+ph+52), 'attempts (k)', font=F(15), fill=BLACK, anchor='mm')
prevb, preva = (X(0), Y(0)), (X(0), Y(0))
for k,v in base:
    d.line([prevb[0], prevb[1], X(k), Y(v)], fill=GRAY, width=5)
    d.ellipse([X(k)-7,Y(v)-7,X(k)+7,Y(v)+7], fill=GRAY, outline=BLACK)
    prevb = (X(k), Y(v))
for k,v in saaga:
    d.line([preva[0], preva[1], X(k), Y(v)], fill=BLUE, width=5)
    d.ellipse([X(k)-7,Y(v)-7,X(k)+7,Y(v)+7], fill=BLUE, outline=BLACK)
    preva = (X(k), Y(v))
# legend (top-right, white backing)
lgx, lgy = ml+pw-230, mt+14
d.rectangle([lgx, lgy, lgx+224, lgy+58], fill='white', outline=(0x90,0x90,0x90))
d.line([lgx+12, lgy+16, lgx+48, lgy+16], fill=GRAY, width=5); d.ellipse([lgx+26-7,lgy+16-7,lgx+26+7,lgy+16+7], fill=GRAY)
d.text((lgx+58, lgy+16), 'Baseline attack loop', font=F(14), fill=BLACK, anchor='lm')
d.line([lgx+12, lgy+42, lgx+48, lgy+42], fill=BLUE, width=5); d.ellipse([lgx+26-7,lgy+42-7,lgx+26+7,lgy+42+7], fill=BLUE)
d.text((lgx+58, lgy+42), 'SAAGA', font=F(14, True), fill=BLACK, anchor='lm')
im.save(f'{OUT}/fig3_topk.png'); print('fig3 ok')

# ---------------------------------------------------------------- Figure 4: per defense family
bar_chart(f'{OUT}/fig4_defense.png', 'Verified success by defense family (n)',
    [('password (419)', (66.1,)), ('translation (236)', (38.6,)), ('roleplay (211)', (44.5,)),
     ('trigger-phrase (78)', (33.3,)), ('conditional (29)', (41.4,)), ('instruction-hiding (17)', (41.2,)),
     ('exception (8)', (50.0,)), ('conversation (2)', (50.0,))],
    [BLUE], 'verified success (%)', width_in=3.4, height_in=3.4, ylim=80, hbar=True, annot=True)

# ---------------------------------------------------------------- Figure 5: memory ablation (two panels)
im, d, W, H = canvas(7.1, 2.9)
d.text((W/2, 28), 'Memory ablation (1,000 runs, identical settings otherwise)', font=F(21, True), fill=BLACK, anchor='mm')
def panel(x0, w_in, title, vals, colors, ymax, unit, labels):
    pw2, ph2 = int(w_in*DPI)-80, H-150
    mt, mb = 70, 70
    ml = 70
    px, py = x0+40, 90
    d.text((px+ (pw2+ml)/2, py-18), title, font=F(18, True), fill=BLACK, anchor='mm')
    d.line([px+ml, py, px+ml, py+ph2], fill=BLACK, width=2)
    d.line([px+ml, py+ph2, px+ml+pw2, py+ph2], fill=BLACK, width=2)
    n = len(vals); per = pw2/n; bw = per*0.5
    for i,(v,c) in enumerate(zip(vals, colors)):
        h = ph2*v/ymax
        x0b = px+ml+per*i+per/2-bw/2
        d.rectangle([x0b, py+ph2-h, x0b+bw, py+ph2], fill=c, outline=(0x80,0x80,0x80))
        d.text((x0b+bw/2, py+ph2-h-14), f'{v}', font=F(16, True), fill=BLACK, anchor='mm')
        d.text((x0b+bw/2, py+ph2+22), labels[i], font=F(15), fill=BLACK, anchor='mm')
    for f in [0.5,1.0]:
        y = py+ph2-ph2*f
        d.text((px+ml-10, y), f'{f*ymax:g}', font=F(13), fill=BLACK, anchor='rm')
panel(0, 3.35, 'Break rate (%)', (92.6, 93.2), [GRAY, BLUE], 100, '%', ('memory off','memory on'))
panel(3.55*DPI, 3.35, 'Avg attempts on success', (4.02, 3.61), [GRAY, BLUE], 5, '', ('memory off','memory on'))
im.save(f'{OUT}/fig5_memory.png'); print('fig5 ok')
print('ALL FIGURES DONE')
