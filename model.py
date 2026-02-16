import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_attn_kernel import _attention



########################################################################################################################

# MLP
class MLP(nn.Module):
    def __init__(self, isencoder=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, 3)  # Output layer: 3 classes
        self.isencoder = isencoder

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.isencoder:
            return x
        else:
            return self.fc3(x)


# Flash attention
class MultiHeadSelfAttention_flash(nn.Module):
    def __init__(self, token_dim, num_heads):
        super(MultiHeadSelfAttention_flash, self).__init__()
        assert token_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.o_proj = nn.Linear(token_dim, token_dim)
        self.F = _attention.apply

    def forward(self, x, coeff_blocks, quartet_matrix):
        batch_size, seq_len, token_dim = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_output = self.F(q, k, v, coeff_blocks, quartet_matrix)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, token_dim)
        return self.o_proj(attn_output)

class FeedForwardNetwork_flash(nn.Module):
    def __init__(self, token_dim, hidden_dim):
        super(FeedForwardNetwork_flash, self).__init__()
        self.fc1 = nn.Linear(token_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, token_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerEncoderBlock_flash(nn.Module):
    def __init__(self, token_dim, num_heads, hidden_dim):
        super(TransformerEncoderBlock_flash, self).__init__()
        self.self_attention = MultiHeadSelfAttention_flash(token_dim, num_heads)
        self.norm1 = nn.LayerNorm(token_dim)
        self.ffn = FeedForwardNetwork_flash(token_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x, coeff_blocks, quartet_matrix):
        x = self.norm1(x + self.self_attention(x, coeff_blocks, quartet_matrix))
        x = self.norm2(x + self.ffn(x))
        return x

# QuartFormer
class QuartFormer(nn.Module):
    def __init__(self, species_num):
        super(QuartFormer, self).__init__()
        # Model hyperparameters
        token_dim=256
        num_heads=16
        hidden_dim=16
        num_layers=3
        num_classes=3

        self.species_num = species_num
        self.mlp_layer = MLP(True)
        self.input_layer = nn.Sequential(nn.Linear(self.species_num + 128, token_dim), nn.ReLU())
        self.layers = nn.ModuleList([TransformerEncoderBlock_flash(token_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, x, coeff_blocks, quartet_matrix):

        batch_size, seq_len, feature_dim = x.shape
        x = x.view(batch_size * seq_len, feature_dim)

        x_mlp = x[:, -256:]
        x_mlp = self.mlp_layer(x_mlp)
        x = torch.cat([x[:, :self.species_num], x_mlp], dim=-1)

        # Input layer
        x = self.input_layer(x)
        x = x.view(batch_size, seq_len, -1)

        # Sparse attention layers
        for layer in self.layers:
            x = layer(x, coeff_blocks, quartet_matrix)
        logits = self.classifier(x)
        return logits


###########################################################################################################################



