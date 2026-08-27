"""
Regenerate fig_hbcc_pipeline.png (HBCC 4-Stage Architecture Pipeline).
Reflects stem_stride=1 fix: stages now run at 32×32 / 16×16 / 8×8 / 4×4.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / 'Report_LaTeX' / 'images'
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT  = '#6C757D'
C_STEM   = '#1C2833'
C_HYBRID = '#873600'
C_CLUST  = '#1A5276'
C_HEAD   = '#1C2833'
C_ARROW  = '#4A4A4A'
C_LABEL  = '#555555'

FW, FH = 14.2, 3.6
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

BOXTOP = 2.35
BOXBOT = 0.90
BOXY   = (BOXTOP + BOXBOT) / 2
BOXH   = BOXTOP - BOXBOT
LABEL_Y = BOXTOP + 0.26   # resolution label y (above boxes)
CONN_Y  = BOXBOT - 0.22   # stride-2 annotation y (below boxes)

# Element x-centres
XS = {
    'input':  0.75,
    'stem':   2.35,
    'stg1':   4.40,
    'stg2':   6.65,
    'stg3':   8.90,
    'stg4':  11.15,
    'head':  13.05,
}

WS = {
    'input': 1.10,
    'stem':  1.40,
    'stg':   1.60,
    'head':  1.20,
}

RADI = 0.06
GAP  = 0.12   # clearance between box edge and arrow tip


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_box(cx, bw, lines, color, fs=9.5):
    x = cx - bw / 2
    r = FancyBboxPatch((x, BOXBOT), bw, BOXH,
                       boxstyle=f'round,pad={RADI}',
                       facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(r)
    if isinstance(lines, str):
        lines = lines.split('\n')
    n = len(lines)
    sp = BOXH / (n + 0.35)
    for i, line in enumerate(lines):
        yo = (n - 1) / 2 * sp - i * sp
        fw = 'bold' if i == 0 else 'normal'
        ax.text(cx, BOXY + yo, line, ha='center', va='center',
                color='white', fontsize=fs, fontweight=fw, zorder=4)


def simple_arrow(x0, x1):
    """Single straight arrow at BOXY level."""
    ax.annotate('', xy=(x1, BOXY), xytext=(x0, BOXY),
                arrowprops=dict(arrowstyle='->', color=C_ARROW,
                                lw=1.6, mutation_scale=12),
                zorder=5)


def res_label(cx, text):
    """Resolution label above a box."""
    ax.text(cx, LABEL_Y, text, ha='center', va='bottom',
            fontsize=8.5, color=C_LABEL, style='italic')


def conn_label(x0, x1, text='stride-2 conv'):
    """Small italic label below a connector arrow."""
    cx = (x0 + x1) / 2
    ax.text(cx, CONN_Y, text, ha='center', va='top',
            fontsize=7.5, color=C_LABEL, style='italic')


# ── Boxes ─────────────────────────────────────────────────────────────────────
draw_box(XS['input'], WS['input'],
         'Input\n32×32\n3 ch', C_INPUT)

draw_box(XS['stem'], WS['stem'],
         'Stem\n(+XY, stride 1)\n64 ch', C_STEM)
res_label(XS['stem'], '32×32')

draw_box(XS['stg1'], WS['stg'],
         'Stage 1\nHybrid + LBPConv\nfold 4×4  ·  64 ch', C_HYBRID)
res_label(XS['stg1'], '32×32')

draw_box(XS['stg2'], WS['stg'],
         'Stage 2\nHybrid + DWConv\nfold 2×2  ·  96 ch', C_HYBRID)
res_label(XS['stg2'], '16×16')

draw_box(XS['stg3'], WS['stg'],
         'Stage 3\nCluster\nfold 1×1  ·  192 ch', C_CLUST)
res_label(XS['stg3'], '8×8')

draw_box(XS['stg4'], WS['stg'],
         'Stage 4\nCluster\nfold 1×1  ·  256 ch', C_CLUST)
res_label(XS['stg4'], '4×4')

draw_box(XS['head'], WS['head'],
         'GAP +\nLinear\n→ Classes', C_HEAD)

# ── Arrows ────────────────────────────────────────────────────────────────────
# Input → Stem
simple_arrow(XS['input'] + WS['input']/2 + GAP,
             XS['stem']  - WS['stem']/2  - GAP)

# Stem → Stage 1  (same resolution, plain arrow)
simple_arrow(XS['stem']  + WS['stem']/2  + GAP,
             XS['stg1']  - WS['stg']/2   - GAP)

# Stage 1 → Stage 2  (stride-2 downsampling)
x0_12 = XS['stg1'] + WS['stg']/2 + GAP
x1_12 = XS['stg2'] - WS['stg']/2 - GAP
simple_arrow(x0_12, x1_12)
conn_label(x0_12, x1_12)

# Stage 2 → Stage 3
x0_23 = XS['stg2'] + WS['stg']/2 + GAP
x1_23 = XS['stg3'] - WS['stg']/2 - GAP
simple_arrow(x0_23, x1_23)
conn_label(x0_23, x1_23)

# Stage 3 → Stage 4
x0_34 = XS['stg3'] + WS['stg']/2 + GAP
x1_34 = XS['stg4'] - WS['stg']/2 - GAP
simple_arrow(x0_34, x1_34)
conn_label(x0_34, x1_34)

# Stage 4 → Head
simple_arrow(XS['stg4'] + WS['stg']/2 + GAP,
             XS['head']  - WS['head']/2 - GAP)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(FW/2, FH - 0.25, 'HBCC — 4-Stage Architecture Pipeline',
        ha='center', va='top', fontsize=13, fontweight='bold', color='#1A252F')

# ── Legend ────────────────────────────────────────────────────────────────────
LEG_Y = 0.42
legend_items = [
    (C_HYBRID, 'Hybrid stage (local + cluster branches)'),
    (C_CLUST,  'Cluster-only stage'),
    (C_STEM,   'Stem / Classifier head'),
]
lx = 1.8
for color, label in legend_items:
    patch = FancyBboxPatch((lx, LEG_Y - 0.14), 0.30, 0.28,
                            boxstyle='round,pad=0.03',
                            facecolor=color, edgecolor='none', zorder=3)
    ax.add_patch(patch)
    ax.text(lx + 0.44, LEG_Y, label, va='center',
            fontsize=8.8, color='#2C3E50')
    lx += 4.1

out = OUT / 'fig_hbcc_pipeline.png'
fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')
