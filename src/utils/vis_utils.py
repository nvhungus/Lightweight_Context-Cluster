import matplotlib.pyplot as plt
import numpy as np

def plot_cosine_vs_hamming(cos_sims, ham_dists, pearson_corr, save_path=None):
    """
    Vẽ biểu đồ so sánh phân phối và tương quan giữa Cosine Similarity và Hamming Distance.
    
    Args:
        cos_sims (np.ndarray): Mảng chứa các giá trị Cosine Similarity.
        ham_dists (np.ndarray): Mảng chứa các giá trị Hamming Distance.
        pearson_corr (float): Hệ số tương quan Pearson.
        save_path (str, optional): Đường dẫn để lưu file ảnh (.png).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Biểu đồ 1: Phân phối (Distributions)
    axes[0].hist(cos_sims, bins=30, alpha=0.5, color='blue', density=True, label='Cosine Sim')
    axes[0].set_xlabel('Cosine Similarity', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distribution of Cosine Sim vs Hamming Dist', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Vẽ trục y thứ 2 cho Hamming distance trên cùng biểu đồ 1
    ax0_twin = axes[0].twiny()
    ax0_twin.hist(ham_dists, bins=30, alpha=0.5, color='red', density=True, label='Hamming Dist')
    ax0_twin.set_xlabel('Hamming Distance', fontsize=12)
    
    # Gom legend
    lines_1, labels_1 = axes[0].get_legend_handles_labels()
    lines_2, labels_2 = ax0_twin.get_legend_handles_labels()
    axes[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center')

    # Biểu đồ 2: Scatter plot chứng minh tương quan tuyến tính
    axes[1].scatter(ham_dists, cos_sims, alpha=0.6, color='purple', edgecolors='k')
    axes[1].set_title(f'Correlation Analysis\nPearson r: {pearson_corr:.4f}', fontsize=14)
    axes[1].set_xlabel('Hamming Distance', fontsize=12)
    axes[1].set_ylabel('Cosine Similarity', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[*] Đã lưu biểu đồ tại: {save_path}")
    plt.show()

def plot_point_cloud_2d(data_np, title, save_path=None):
    """
    Hỗ trợ vẽ Point Cloud 2D (Dành cho Notebook 02 sau này).
    """
    h, w, c = data_np.shape
    data_reshaped = data_np.reshape(-1, 3)
    
    y, x = np.mgrid[0:h, 0:w]
    x = x.flatten()
    y = -y.flatten() # Đảo trục y
    
    colors = np.clip(data_reshaped, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c=colors, s=15, alpha=0.9, marker='o')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()