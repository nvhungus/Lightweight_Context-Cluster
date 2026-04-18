import matplotlib.pyplot as plt
import numpy as np

def plot_point_cloud_comparison(coords_before, colors_before, coords_after, colors_after, titles, save_path=None):
    """Vẽ so sánh 2 Point Clouds cạnh nhau (Trước và sau khi Shrink)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    datasets = [
        (coords_before, colors_before, titles[0]),
        (coords_after, colors_after, titles[1])
    ]
    
    for i, (coords, colors, title) in enumerate(datasets):
        # Normalize màu nếu cần
        if colors.max() > 1.0 or colors.min() < 0.0:
            colors = (colors - colors.min()) / (colors.max() - colors.min())
            
        axes[i].scatter(coords[:, 0], coords[:, 1], c=colors.numpy(), s=20 if i==0 else 80)
        axes[i].set_title(title)
        axes[i].set_aspect('equal')
        axes[i].axis('off')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()