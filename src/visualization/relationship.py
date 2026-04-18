import matplotlib.pyplot as plt

def plot_correlation_scatter(x, y, xlabel, ylabel, title, subtitle=None, save_path=None):
    """Vẽ biểu đồ phân tán để xem tương quan (VD: Hamming vs Cosine)"""
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6, color='purple', edgecolors='k')
    
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=12, color='gray')
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_manifold_collapse_comparison(X_real, X_relu, X_linear, labels, save_path=None):
    """Vẽ 3 biểu đồ scatter cạnh nhau để so sánh Manifold Collapse (Notebook 03)"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.bwr
    
    datasets = [
        (X_real, '1. Original Features (Real-valued)', 'black'),
        (X_relu, '2. With ReLU + Sign\n(MANIFOLD COLLAPSE!)', 'red'),
        (X_linear, '3. Linear Bottleneck + Sign\n(Structure Preserved)', 'green')
    ]
    
    for i, (data, title, color) in enumerate(datasets):
        axes[i].scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, edgecolors='k', alpha=0.5, s=50)
        axes[i].set_title(title, fontsize=14, color=color, pad=15)
        axes[i].set_xlim(-6 if i==0 else -1.5, 6 if i==0 else 1.5)
        axes[i].set_ylim(-6 if i==0 else -1.5, 6 if i==0 else 1.5)
        
        if i > 0:
            axes[i].set_xticks([-1, 0, 1])
            axes[i].set_yticks([-1, 0, 1])
            
    # Annotations
    axes[1].annotate('All points collapsed\ninto (1, 1)', xy=(1, 1), xytext=(-0.8, -0.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8), fontsize=12, color='red', weight='bold')

    if save_path:
        plt.savefig(save_path)
    plt.show()