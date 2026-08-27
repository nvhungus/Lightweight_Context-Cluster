"""
Regenerate Pareto charts for CIFAR-10, Tiny-ImageNet-200, and Food-101.
All numbers verified from benchmark JSONs and metrics.jsonl files (August 2026).

Each small-image figure is ~2 in tall so two stack in one IEEE column.
Food-101 is a single-panel (params only) chart because latency was not
measured at 224x224 in the same hardware conditions.

CIFAR-10 / TIN: CE models were trained with an older architecture
(depths=[2,2,4,2], stem_stride=2). KD models use the current architecture
(depths=[1,1,2,1], stem_stride=1). CE and KD are architecturally comparable
only on TIN where both regimes share the identical architecture.

Food-101: only HBCC-2.5M and ResNet-18 are compared (CE-only protocol).
CIFAR-100 data is retained below for reference but no figure is produced;
Food-101 replaces CIFAR-100 in the paper.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# Save to both Vietnamese and English image directories.
ROOT = Path(__file__).parent.parent / 'Report_LaTeX'
OUT_DIRS = [ROOT / 'images', ROOT / 'en' / 'images']
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.linewidth': 0.75,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
})

# ── Style catalogue ──────────────────────────────────────────────────────────
# (color, marker, markersize, filled, zorder, display_label)
STYLE = {
    'ResNet-18':    ('#3D3D3D', 's',  6.5, True,  2, 'ResNet-18'),
    'HBCC-Med+KD': ('#7B241C', '*', 12.0, True,  6, 'HBCC-Med+KD'),
    'HBCC-Sml+KD': ('#C0392B', '*', 10.0, True,  6, 'HBCC-Sml+KD'),
    'HBCC-Med':    ('#7B241C', 'o',  6.5, False, 4, 'HBCC-Med (CE)'),
    'HBCC-Sml':    ('#C0392B', 'o',  6.5, False, 4, 'HBCC-Sml (CE)'),
    'ShuffleNet':  ('#1A5276', '^',  6.5, True,  3, 'ShuffleNetV2-x1.0'),
    'MobileNet':   ('#1E8449', 'D',  6.0, True,  3, 'MobileNetV2'),
    'CoC':         ('#6C3483', 'P',  6.5, True,  3, 'CoC-baseline'),
    'HBCC-25M':    ('#7B241C', '*', 12.0, True,  6, 'HBCC-2.5M'),
}

# ── Verified data: (key, top1_acc%, params_M, latency_ms_b1) ────────────────

# CIFAR-10 — KD models + baselines only (CE excluded: different architecture)
CIFAR10 = [
    ('ResNet-18',    96.48, 11.17,  3.03),
    ('MobileNet',    91.04,  2.24,  5.33),
    ('ShuffleNet',   91.68,  1.26,  6.53),
    ('CoC',          88.34,  2.40, 12.47),
    ('HBCC-Sml+KD', 94.93,  1.27,  8.41),
    ('HBCC-Med+KD', 95.36,  1.76,  8.09),
]

# Tiny-ImageNet — CE and KD share identical architecture → arrows valid
TIN = [
    ('ResNet-18',    58.58, 11.27,  3.13),
    ('MobileNet',    49.97,  2.48,  4.66),
    ('ShuffleNet',   55.68,  1.46,  6.04),
    ('CoC',          49.30,  2.46, 11.52),
    ('HBCC-Sml',    56.15,  1.32,  7.77),
    ('HBCC-Med',    56.80,  1.80,  8.20),
    ('HBCC-Sml+KD', 57.56,  1.32,  7.50),
    ('HBCC-Med+KD', 58.90,  1.80,  7.57),
]

# Food-101 — HBCC-2.5M vs ResNet-18, CE-only, 224×224.
# Only params available; latency not measured in the same hardware conditions.
# (key, top1_acc%, params_M)
FOOD101 = [
    ('ResNet-18', 82.91, 11.228),
    ('HBCC-25M',  82.53,  2.566),
]

# CIFAR-100 data retained for reference (no figure produced in this paper).
CIFAR100 = [
    ('ResNet-18',    79.05, 11.22,  3.03),
    ('ShuffleNet',   76.46,  1.36,  5.77),
    ('HBCC-Med+KD', 77.68,  1.78,  7.83),
    ('HBCC-Sml+KD', 76.75,  1.30,  7.51),
    ('HBCC-Med',    75.59,  1.78,  7.69),
    ('HBCC-Sml',    75.40,  1.30,  7.59),
    ('MobileNet',    73.75,  2.35,  4.77),
    ('CoC',          72.01,  2.42, 11.21),
]

# CE→KD arrows drawn only on TIN params panel
TIN_ARROWS = [('HBCC-Med', 'HBCC-Med+KD'), ('HBCC-Sml', 'HBCC-Sml+KD')]

# Fixed legend marker sizes — chosen to fit inside the 4.8–5.0 pt font legend box.
_LEGEND_MS = {'*': 5.0, '^': 4.5, 'D': 4.2, 'P': 4.5, 's': 4.5, 'o': 4.5}


def _legend_handles(raw_handles):
    """Return Line2D proxy artists with fixed small sizes for the legend."""
    out = []
    for h in raw_handles:
        mk = h.get_marker()
        out.append(Line2D([0], [0], color=h.get_color(), marker=mk,
                          markersize=_LEGEND_MS.get(mk, 4.5),
                          mfc=h.get_markerfacecolor(), mew=0.9, ls='none'))
    return out


def _find(data, key):
    for row in data:
        if row[0] == key:
            return row
    return None


def panel(ax, data, xcol, xlog, ylabel, arrows=None):
    for key, acc, params, lat in data:
        c, mk, ms, filled, zo, lbl = STYLE[key]
        x = params if xcol == 'params' else lat
        mfc = c if filled else 'white'
        ax.plot(x, acc, color=c, marker=mk, ms=ms,
                mfc=mfc, mew=0.9, ls='none', zorder=zo, label=lbl)

    if xcol == 'params' and arrows:
        for key_ce, key_kd in arrows:
            r_ce = _find(data, key_ce)
            r_kd = _find(data, key_kd)
            if r_ce and r_kd:
                ax.annotate('', xy=(r_kd[2], r_kd[1]),
                            xytext=(r_ce[2], r_ce[1]),
                            arrowprops=dict(arrowstyle='->',
                                            color='#922B21',
                                            lw=0.9, mutation_scale=8))

    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, ls='--', alpha=0.32, lw=0.4)
    ax.tick_params(labelsize=6.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def make(data, stem, suffix, arrows=None):
    """Two-panel figure (params + latency) for small-image datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.10))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.93,
                        bottom=0.32, wspace=0.44)

    panel(ax1, data, 'params', xlog=True,
          ylabel='Top-1 Acc. (%)', arrows=arrows)
    ax1.set_xlabel('Params (M, log)', fontsize=6.5)
    ax1.set_title(suffix, fontsize=7, pad=2)

    panel(ax2, data, 'lat', xlog=False, ylabel='', arrows=None)
    ax2.set_xlabel('Latency b=1 (ms)', fontsize=6.5)

    handles, labels = ax1.get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); uh.append(h); ul.append(l)

    fig.legend(_legend_handles(uh), ul, ncol=4, fontsize=4.8,
               loc='lower center', bbox_to_anchor=(0.57, 0.01),
               framealpha=0.92, edgecolor='#CCCCCC',
               handlelength=0.9, columnspacing=0.4,
               borderpad=0.4, labelspacing=0.25,
               handletextpad=0.3)

    for d in OUT_DIRS:
        for ext in ('png', 'pdf'):
            fig.savefig(d / f'{stem}.{ext}', dpi=250,
                        bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved → {stem}.(png|pdf)')


def make_food101(data, stem, suffix):
    """Single-panel params-only chart for Food-101.

    Latency was not measured in the same hardware conditions as the
    small-image benchmarks, so only the parameter-efficiency axis is shown.
    """
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.10))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.32)

    for key, acc, params in data:
        c, mk, ms, filled, zo, lbl = STYLE[key]
        mfc = c if filled else 'white'
        ax.plot(params, acc, color=c, marker=mk, ms=ms,
                mfc=mfc, mew=0.9, ls='none', zorder=zo, label=lbl)

    ax.set_xscale('log')
    ax.set_xlim(1.8, 16.0)
    ax.set_ylim(82.1, 83.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(True, ls='--', alpha=0.32, lw=0.4)
    ax.tick_params(labelsize=6.5)
    ax.set_ylabel('Top-1 Acc. (%)', fontsize=7)
    ax.set_xlabel('Params (M, log)', fontsize=6.5)
    ax.set_title(suffix, fontsize=7, pad=2)

    # Accuracy annotations placed to avoid overlap on the narrow y range.
    for key, acc, params in data:
        c = STYLE[key][0]
        if key == 'ResNet-18':
            # Label above-left of the point (towards the chart interior)
            ax.text(params * 0.58, acc + 0.10, f'{acc:.2f}%',
                    fontsize=5.5, color=c, ha='center', va='bottom')
        else:
            # Label below-right of the point (towards the chart interior)
            ax.text(params * 1.65, acc - 0.10, f'{acc:.2f}%',
                    fontsize=5.5, color=c, ha='center', va='top')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(_legend_handles(handles), labels, ncol=2, fontsize=5.0,
               loc='lower center', bbox_to_anchor=(0.55, 0.01),
               framealpha=0.92, edgecolor='#CCCCCC',
               handlelength=0.9, columnspacing=0.5,
               borderpad=0.4, labelspacing=0.25,
               handletextpad=0.3)

    for d in OUT_DIRS:
        for ext in ('png', 'pdf'):
            fig.savefig(d / f'{stem}.{ext}', dpi=250,
                        bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    print(f'Saved → {stem}.(png|pdf)')


make(CIFAR10, 'pareto_cifar10', 'CIFAR-10')
make(TIN,     'pareto_tin',     'Tiny-ImageNet-200', arrows=TIN_ARROWS)
make_food101(FOOD101, 'pareto_food101', 'Food-101')
print('Done.')