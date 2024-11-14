import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde

def adaptive_bandwidth(e, neighbors, base_bandwidth=1.0):
    """
    Calculate an adaptive bandwidth based on the distance to nearby embeddings.
    :param e: The current embedding.
    :param neighbors: The neighboring embeddings.
    :param base_bandwidth: The base bandwidth to scale.
    :return: Adaptive bandwidth value.
    """
    distances = np.linalg.norm(neighbors - e, axis=1)
    avg_distance = np.mean(distances)
    return base_bandwidth * avg_distance

def density_estimation(embeddings, bandwidths):
    """
    Apply KDE to estimate the density of each embedding.
    :param embeddings: List of embeddings to calculate density for.
    :param bandwidths: List of bandwidths for each embedding.
    :return: List of density values.
    """
    densities = []
    for i, e in enumerate(embeddings):
        kde = gaussian_kde(embeddings.T, bw_method=bandwidths[i])
        densities.append(kde(e))
    return densities

def detect_rare_species_embeddings(embeddings, density_threshold_percentile=0.05, base_bandwidth=1.0):
    """
    Detect rare species embeddings based on clustering and KDE density estimation.
    :param embeddings: The set of all species embeddings.
    :param density_threshold_percentile: Percentile threshold for determining low-density embeddings.
    :param base_bandwidth: Base bandwidth for KDE.
    :return: List of embeddings detected as rare species.
    """
    # Step 1: Perform clustering using GMM
    gmm = GaussianMixture(n_components=5, random_state=0)
    cluster_labels = gmm.fit_predict(embeddings)
    cluster_probs = gmm.predict_proba(embeddings)

    # Identify low-density clusters
    low_density_clusters = []
    for cluster in range(gmm.n_components):
        if np.mean(cluster_probs[:, cluster]) < density_threshold_percentile:
            low_density_clusters.append(cluster)

    # Select embeddings in low-density clusters
    low_density_embeddings = embeddings[np.isin(cluster_labels, low_density_clusters)]

    # Step 2: KDE-Based Density Estimation
    densities = []
    bandwidths = []
    for e in low_density_embeddings:
        # Calculate adaptive bandwidth for each embedding
        neighbors = low_density_embeddings  # Assuming all in low-density cluster are neighbors
        bandwidth = adaptive_bandwidth(e, neighbors, base_bandwidth)
        bandwidths.append(bandwidth)

    # Estimate densities for each embedding in low-density clusters
    densities = density_estimation(low_density_embeddings, bandwidths)

    # Apply threshold to identify rare species
    density_threshold = np.percentile(densities, density_threshold_percentile * 100)
    rare_species_embeddings = low_density_embeddings[np.array(densities) < density_threshold]

    return rare_species_embeddings

# Example usage
# Assume `embeddings` is an array of shape (n_samples, embedding_dim)
# embeddings = np.random.rand(100, 64)  # Example embeddings
# rare_species_embeddings = detect_rare_species_embeddings(embeddings)
# print("Rare species embeddings:", rare_species_embeddings)
