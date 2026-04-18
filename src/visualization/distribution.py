import matplotlib.pyplot as plt

def plot_dual_histogram(data1, data2, label1, label2, xlabel1, xlabel2, title, save_path=None):
    """Vẽ 2 histogram lồng nhau với 2 trục X khác biệt (VD: Cosine vs Hamming)"""
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Vẽ data1
    ax1.hist(data1, bins=30, alpha=0.5, color='blue', density=True, label=label1)
    ax1.set_xlabel(xlabel1, color='blue', weight='bold')
    ax1.set_ylabel('Density')
    ax1.tick_params(axis='x', colors='blue')

    # Vẽ data2 trên trục X thứ 2
    ax2 = ax1.twiny()
    ax2.hist(data2, bins=30, alpha=0.5, color='red', density=True, label=label2)
    ax2.set_xlabel(xlabel2, color='red', weight='bold')
    ax2.tick_params(axis='x', colors='red')

    ax1.set_title(title, pad=20)
    
    # Gom legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

    if save_path:
        plt.savefig(save_path)
    plt.show()