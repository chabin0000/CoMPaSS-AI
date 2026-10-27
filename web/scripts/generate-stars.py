"""Build-time only: fixed-seed targets, spatial correspondence, progressive density.
No research code or runtime font dependency. Requires Pillow, available in bundled Python.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math, random, struct, json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public' / 'universe'
OUT.mkdir(parents=True, exist_ok=True)
N = 12000
rng = random.Random(20261027)
font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 270)

def word_points(word):
    widths = [font.getlength(c) for c in word]
    width = math.ceil(sum(widths) + (len(word)-1)*24 + 20)
    im = Image.new('L', (width, 340))
    draw = ImageDraw.Draw(im)
    x = 10
    for c, w in zip(word, widths):
        draw.text((x, 0), c, font=font, fill=255, stroke_width=0)
        x += w+24
    bbox = im.getbbox()
    mask = im.load()
    pts = []
    while len(pts) < N:
        x, y = rng.uniform(bbox[0], bbox[2]), rng.uniform(bbox[1], bbox[3])
        if mask[int(x), int(y)] > 200:
            # Occasional stardust, but counters and outlines remain legible.
            dust = rng.random() < .09
            dx = rng.gauss(0, 3.6) if dust else 0
            dy = rng.gauss(0, 3.6) if dust else 0
            pts.append(((x+dx-width/2)/width*1.72, ((bbox[1]+bbox[3])/2-y+dy)/width*1.72, rng.gauss(0,.027)))
    return pts

def cluster(i):
    side = -1 if i < N/2 else 1
    r = .31 * rng.random()**.62
    a = (i % 2)*math.pi + r*14 + rng.gauss(0,.6)
    return (side*.43 + math.cos(a)*r, math.sin(a)*r*.78, rng.gauss(0,.075))

def verification(i):
    if i < N*.66:
        a = rng.uniform(0, math.tau)
        r = rng.gauss(.295,.009)
        return (math.cos(a)*r, math.sin(a)*r, rng.gauss(0,.012))
    t = rng.random()
    if t < .38:
        q = t/.38
        x,y = -.13+.11*q, .005-.105*q
    else:
        q = (t-.38)/.62
        x,y = -.02+.19*q, -.10+.23*q
    return (x+rng.gauss(0,.011), y+rng.gauss(0,.011), rng.gauss(0,.02))

def student(i):
    if i % 4 == 0:
        a = rng.uniform(0,math.tau)
        x,y = .34*math.cos(a), .105*math.sin(a)
        return (x*.94-y*.34+rng.gauss(0,.003), x*.34+y*.94+rng.gauss(0,.003), rng.gauss(0,.02))
    r = min(.19, abs(rng.gauss(0,.075)))
    a = rng.uniform(0,math.tau)
    return (r*math.cos(a), r*math.sin(a), rng.gauss(0,.06))

def edges(_):
    side = rng.choice((-1,1))
    return (side*rng.uniform(.79,1.32), rng.uniform(-1.6,1.6), rng.uniform(-.12,.12))

def spread(_):
    a,r = rng.uniform(0,math.tau), rng.uniform(.10,1.4)
    return (math.cos(a)*r, math.sin(a)*r*.7, rng.uniform(-.16,.16))

def morton(p):
    x = max(0,min(65535,int((p[0]+1.7)/3.4*65535)))
    y = max(0,min(65535,int((p[1]+1.7)/3.4*65535)))
    v=0
    for b in range(16):
        v |= ((x>>b)&1)<<(2*b)
        v |= ((y>>b)&1)<<(2*b+1)
    return v

targets = [word_points('ARGOS'),[cluster(i) for i in range(N)],[verification(i) for i in range(N)],
           [student(i) for i in range(N)],[edges(i) for i in range(N)],[spread(i) for i in range(N)],word_points('COMPASS')]
# Spatial ordering avoids all-to-all crossing. A single shared permutation gives
# every low-density prefix coverage of every letter and every target.
targets = [sorted(t,key=morton) for t in targets]
order = list(range(N))
rng.shuffle(order)
for index, target in enumerate(targets):
    (OUT/f'shape-{index}.bin').write_bytes(struct.pack('<'+'f'*(N*3),*[v for i in order for v in target[i]]))
    if index in (0,1,2,3,6):
        circles=[]
        for i in order[:2600]:
            x,y,_=target[i]
            r = rng.choices([.65,.95,1.4,2.1],[.44,.38,.16,.02])[0]
            color = rng.choices(['#adc6e0','#eaf7ff','#f1c9a7'],[.42,.55,.03])[0]
            circles.append(f'<circle cx="{500+x*500:.1f}" cy="{220-y*500:.1f}" r="{r}" fill="{color}"/>')
        (OUT/f'static-{index}.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 440">'+''.join(circles)+'</svg>',encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({'count':N,'shapes':7,'seed':20261027,'format':'little-endian Float32 xyz; spatial Morton match, shared random progressive order','text':['ARGOS','COMPASS']}),encoding='utf-8')
print(f'Generated {len(targets)} targets × {N} points, plus static fallbacks.')
