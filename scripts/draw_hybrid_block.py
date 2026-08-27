"""
Regenerate fig_hybrid_block.png — HybridClusterBlock residual block diagram.
All boxes and text fully visible, no clipping.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / 'Report_LaTeX' / 'images'
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK   = '#1C2833'   # input / output
C_GRAY   = '#5D6D7E'   # norm layers
C_BLUE   = '#1A6FA0'   # cluster branch
C_ORANGE = '#873600'   # local branch
C_TEAL   = '#148F77'   # fuse
C_DARK2  = '#17202A'   # residual-add boxes
C_NAVY   = '#1B4F72'   # MLP
C_SKIP1  = '#7F8C8D'   # first skip line
C_SKIP2  = '#AAB7B8'   # second skip line

FW, FH = 6.5, 11.2
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('white')

CX   = FW / 2          # main column centre  x = 3.25
BW   = 4.30            # standard box width
SBW  = 1.90            # side-branch box width
BH   = 0.60            # single-line box height
TBH  = 1.05            # three-line branch box height
RADI = 0.06            # corner radius pad


# ── Helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, lines, bg, fs=9.5, bold_first=False):
    """Draw rounded-rect + centered multi-line text."""
    r = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f'round,pad={RADI}',
        facecolor=bg, edgecolor='none', zorder=3)
    ax.add_patch(r)
    if isinstance(lines, str):
        lines = lines.split('\n')
    n = len(lines)
    spacing = h / (n + 0.35)
    for i, line in enumerate(lines):
        yo = (n - 1) / 2 * spacing - i * spacing
        fw = 'bold' if (bold_first and i == 0) else 'normal'
        ax.text(cx, cy + yo, line, ha='center', va='center',
                color='white', fontsize=fs, fontweight=fw, zorder=4)


def darrow(x, y0, y1, color='#2C3E50'):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                mutation_scale=13), zorder=5)


def hline(x0, x1, y, c='#2C3E50', ls='-', lw=1.4):
    ax.plot([x0, x1], [y, y], c=c, ls=ls, lw=lw, zorder=2)


def vline(x, y0, y1, c='#2C3E50', ls='-', lw=1.4):
    ax.plot([x, x], [y0, y1], c=c, ls=ls, lw=lw, zorder=2)


def skip_arrow(rx, y_target, c):
    ax.annotate('', xy=(CX + BW / 2 + 0.06, y_target), xytext=(rx, y_target),
                arrowprops=dict(arrowstyle='->', color=c, lw=1.2,
                                mutation_scale=10), zorder=3)


# ── Y positions (inches from bottom) ─────────────────────────────────────────
y_title  = 10.80
y_input  = 10.10
y_norm1  =  9.18
y_slabel =  8.55      # "channel split by ratio ρ" text
y_branch =  7.55      # branch boxes centre
y_merge  =  6.75      # merge point after branches
y_fuse   =  6.12
y_add1   =  5.10
y_norm2  =  4.28
y_mlp    =  3.46
y_add2   =  2.42
y_output =  1.38

RX1 = FW - 0.38      # right-side x for first skip line
RX2 = FW - 0.80      # right-side x for second skip line (inset slightly)
LCX = CX - 1.38      # left branch centre
RCX = CX + 1.38      # right branch centre

# ── Drawing ───────────────────────────────────────────────────────────────────
# Title
ax.text(CX, y_title, 'HybridClusterBlock — Residual Block Structure',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1A252F', zorder=4)

# 1. Input
box(CX, y_input, BW, BH,
    r'$\mathbf{Input}\ \ x\ \ (B,\,C,\,H,\,W)$',
    C_DARK, fs=10.5, bold_first=False)

darrow(CX, y_input - BH/2, y_norm1 + BH/2)

# 2. Norm1
box(CX, y_norm1, BW, BH,
    r'$\mathbf{Norm_1}$  (BatchNorm)',
    C_GRAY, fs=10.0)

darrow(CX, y_norm1 - BH/2, y_slabel + 0.09)

# 3. Split label
ax.text(CX, y_slabel, r'channel split by ratio  $\rho$',
        ha='center', va='center', fontsize=9, color='#7F8C8D',
        style='italic', zorder=4)

# Lines: norm1 → split point → two branches
split_y = y_slabel - 0.22
vline(CX, y_slabel - 0.11, split_y)
hline(LCX, RCX, split_y)
darrow(LCX, split_y, y_branch + TBH/2 + 0.05)
darrow(RCX, split_y, y_branch + TBH/2 + 0.05)

# 4. Cluster branch
box(LCX, y_branch, SBW, TBH,
    'Cluster Branch\n'
    r'$C_c = C(1-\rho)$ ch' + '\nContextClusterOp',
    C_BLUE, fs=9.0, bold_first=True)

# 5. Local branch
box(RCX, y_branch, SBW, TBH,
    'Local Branch\n'
    r'$C_l = C\cdot\rho$ ch' + '\nLBPConv / DWConv',
    C_ORANGE, fs=9.0, bold_first=True)

# Merge: two branches → single line → fuse box
vline(LCX, y_branch - TBH/2, y_merge)
vline(RCX, y_branch - TBH/2, y_merge)
hline(LCX, RCX, y_merge)
darrow(CX, y_merge, y_fuse + BH/2 + 0.05)

# 6. Fuse
box(CX, y_fuse, BW + 0.30, BH,
    'cat  →  Channel Shuffle  →  1×1 Conv  (Fuse)',
    C_TEAL, fs=9.5)

darrow(CX, y_fuse - BH/2, y_add1 + BH/2 + 0.05)

# 7. Layer-scale residual 1
box(CX, y_add1, BW, BH,
    r'$x \;\leftarrow\; x + \mathrm{DropPath}(\boldsymbol{\gamma}_1 \odot \Delta_y)$'
    r'  — layer-scale',
    C_DARK2, fs=8.8)

darrow(CX, y_add1 - BH/2, y_norm2 + BH/2 + 0.05)

# 8. Norm2
box(CX, y_norm2, BW, BH,
    r'$\mathbf{Norm_2}$  (BatchNorm)',
    C_GRAY, fs=10.0)

darrow(CX, y_norm2 - BH/2, y_mlp + BH/2 + 0.05)

# 9. MLP
box(CX, y_mlp, BW + 0.30, BH,
    r'MLP  (FC $\to$ GELU $\to$ FC,   expansion ratio  $r\!=\!3$)',
    C_NAVY, fs=9.5)

darrow(CX, y_mlp - BH/2, y_add2 + BH/2 + 0.05)

# 10. Layer-scale residual 2
box(CX, y_add2, BW, BH,
    r'$x \;\leftarrow\; x + \mathrm{DropPath}(\boldsymbol{\gamma}_2 \odot \Delta_{\mathrm{MLP}})$'
    r'  — layer-scale',
    C_DARK2, fs=8.8)

darrow(CX, y_add2 - BH/2, y_output + BH/2 + 0.05)

# 11. Output
box(CX, y_output, BW * 0.52, BH,
    r'$\mathbf{Output}\ \ x$',
    C_DARK, fs=10.5)

# ── Skip connection 1: Input x → add1 ────────────────────────────────────────
# horizontal leg at y_input
hline(CX + BW/2, RX1, y_input, c=C_SKIP1, ls='--')
# vertical leg down to y_add1
vline(RX1, y_input, y_add1, c=C_SKIP1, ls='--')
# arrow into add1 box
skip_arrow(RX1, y_add1, C_SKIP1)
# label
ax.text(RX1 + 0.07, (y_input + y_add1) / 2, r'$x$',
        ha='left', va='center', fontsize=10, color=C_SKIP1, zorder=4)

# ── Skip connection 2: x₁ (after add1) → add2 ────────────────────────────────
# branch from the mid-point between add1 and norm2
mid12_y = (y_add1 - BH/2 + y_norm2 + BH/2) / 2
hline(CX + BW/2, RX2, mid12_y, c=C_SKIP2, ls='--')
vline(RX2, mid12_y, y_add2, c=C_SKIP2, ls='--')
skip_arrow(RX2, y_add2, C_SKIP2)
ax.text(RX2 + 0.07, (mid12_y + y_add2) / 2, r'$x_1$',
        ha='left', va='center', fontsize=10, color=C_SKIP2, zorder=4)

# ── Save ─────────────────────────────────────────────────────────────────────
out = OUT / 'fig_hybrid_block.png'
fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')
