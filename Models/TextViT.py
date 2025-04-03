import torch
import torch.nn as nn

class TextViT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0.1,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, embedding_dim))

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                      num_heads=num_heads,
                                      mlp_size=mlp_size,
                                      attn_dropout=attn_dropout,
                                      mlp_dropout=mlp_dropout)
              for _ in range(num_transformer_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding[:, :x.size(1), :]
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_size: int = 3072,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0):
        super().__init__()

        # Create MSA block (equation2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # Create MLP block (equation3)
        self.mlp_block = MultiLayerPerceptronLayer(embedding_dim=embedding_dim,
                                                   mlp_size=mlp_size,
                                                   dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 128,
                 num_heads: int = 12,
                 attn_dropout: float = 0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,  # self attention은 qkv모두 같은 vector
                                             key=x,
                                             value=x,
                                             need_weights=False)

        return attn_output


class MultiLayerPerceptronLayer(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
