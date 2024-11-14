import torch
import torch.nn.functional as F

class SENet(torch.nn.Module):
    def __init__(self, embedding_dim, reduction_ratio=16):
        super(SENet, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, embedding_dim // reduction_ratio)
        self.fc2 = torch.nn.Linear(embedding_dim // reduction_ratio, embedding_dim)

    def forward(self, x):
        # Global average pooling
        x = x.mean(dim=-1, keepdim=True)
        # Squeeze and excitation
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def compute_attention_score(embedding, senet):
    """
    Compute attention score for an embedding using SENet.
    :param embedding: The embedding vector.
    :param senet: The Squeeze-and-Excitation network.
    :return: Attention score.
    """
    return senet(embedding)

def compute_gating_vector(embedding, W_gate):
    """
    Compute gating vector for an embedding.
    :param embedding: The embedding vector.
    :param W_gate: The learned weight matrix for gating.
    :return: Gating vector after applying sigmoid activation.
    """
    return torch.sigmoid(W_gate @ embedding)

def advanced_embedding_fusion(E_rare, E_set, senet, W_gate):
    """
    Advanced embedding fusion with attention and gating mechanisms.
    :param E_rare: Rare species embedding.
    :param E_set: Set of other embeddings.
    :param senet: Squeeze-and-Excitation network.
    :param W_gate: Learned weight matrix for gating.
    :return: Fused embedding.
    """
    # Step 1: Compute self-attention score for the rare embedding
    s_rare = compute_attention_score(E_rare, senet)

    # Step 2: Compute attention scores for other embeddings
    s_set = [compute_attention_score(E_j, senet) for E_j in E_set]

    # Step 3: Normalize all attention scores
    scores = torch.cat([s_rare] + s_set, dim=0)
    attention_weights = F.softmax(scores, dim=0)
    alpha_rare = attention_weights[0]
    alpha_set = attention_weights[1:]

    # Step 4: Compute gating vector for the rare embedding
    g_rare = compute_gating_vector(E_rare, W_gate)

    # Step 5: Compute gating vectors for other embeddings
    g_set = [compute_gating_vector(E_j, W_gate) for E_j in E_set]

    # Step 6: Apply gating weights
    E_rare_gated = g_rare * E_rare
    E_set_gated = [g_j * E_j for g_j, E_j in zip(g_set, E_set)]

    # Step 7: Fuse embeddings using attention and gating weights
    E_fused = alpha_rare * E_rare_gated + sum(alpha_j * E_j_gated for alpha_j, E_j_gated in zip(alpha_set, E_set_gated))

    return E_fused

# Example usage
embedding_dim = 64
E_rare = torch.randn(embedding_dim)  # Rare species embedding
E_set = [torch.randn(embedding_dim) for _ in range(5)]  # Set of other embeddings

# Define SENet and learned weight matrix W_gate
senet = SENet(embedding_dim)
W_gate = torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim))

# Compute fused embedding
E_fused = advanced_embedding_fusion(E_rare, E_set, senet, W_gate)
print("Fused embedding:", E_fused)
