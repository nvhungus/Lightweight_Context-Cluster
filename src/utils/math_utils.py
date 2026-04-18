import numpy as np

def generate_binary_vectors(n_samples=1000, dim=256, seed=42):
    """Tạo tập vector nhị phân ngẫu nhiên {-1, 1}."""
    np.random.seed(seed)
    real_vectors = np.random.randn(n_samples, dim)
    binary_vectors = np.sign(real_vectors)
    binary_vectors[binary_vectors == 0] = 1 # Xử lý edge case
    return binary_vectors

def compute_cosine_sim_batch(anchor, batch):
    """Tính Cosine Similarity giữa 1 vector anchor và 1 batch vectors."""
    dot_products = np.dot(batch, anchor)
    norm_anchor = np.linalg.norm(anchor)
    norm_batch = np.linalg.norm(batch, axis=1)
    return dot_products / (norm_anchor * norm_batch)

def compute_hamming_dist_batch(anchor, batch):
    """Tính khoảng cách Hamming (số bit khác nhau)."""
    return np.sum(anchor != batch, axis=1)