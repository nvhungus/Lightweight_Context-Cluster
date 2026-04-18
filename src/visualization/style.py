import matplotlib.pyplot as plt
import seaborn as sns

def set_paper_style():
    """Thiết lập format biểu đồ chuẩn cho Research Paper (IEEE/CVPR style)"""
    sns.set_theme(style="whitegrid")
    
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.format": "png",
        "lines.linewidth": 2.0,
        "lines.markersize": 6
    })