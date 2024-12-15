import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        ffn_dim = d_model // 2
        # Feed-forward network definition
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, q, k, v):
        # Input shape transformation for Conv1d
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # Initial queries, keys and values
        query = self.query_conv(q)
        key = self.key_conv(k)
        value = self.value_conv(v)

        # Apply layer normalization
        query_norm = self.layer_norm1(query.permute(0, 2, 1))
        key_norm = self.layer_norm1(key.permute(0, 2, 1))
        value_norm = self.layer_norm1(value.permute(0, 2, 1))

        # Calculate attention scores
        scores = torch.matmul(query_norm, key_norm.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_norm.size(-1), dtype=torch.float32))
        attn = self.softmax(scores)

        # Compute the output with attention applied to values
        out = torch.matmul(attn, value_norm)

        v = v.permute(0, 2, 1)
        # Adding residual connection for attention output
        out += v  # Adding residual connection
        out = self.layer_norm2(out)  # Apply second layer normalization after residual connection

        # Feed-forward network with residual connection
        ffn_out = self.ffn(out)
        ffn_out += out  # Adding residual connection for FFN
        ffn_out = self.layer_norm2(ffn_out)  # Normalize after FFN

        return ffn_out


class Model(nn.Module):
    def __init__(self, in_channels, num_class, hidden_dim=128, layers=4, ratio= 8):
        super(Model, self).__init__()

        embedding_dim = in_channels // ratio
        self.visual_embedding = nn.Sequential(
            nn.Conv1d(in_channels, embedding_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(embedding_dim * 2, embedding_dim, kernel_size=1),
        )

        self.cross_attention_fp_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])
        self.cross_attention_tp_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])
        self.cross_attention_fp_fuse_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])


        self.tp_cross_attention_fp_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])
        self.tp_cross_attention_tp_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])
        self.tp_cross_attention_fp_fuse_layers = nn.ModuleList([CrossAttention(embedding_dim) for _ in range(layers)])
        
        self.fpv_pose_projection = nn.Conv1d(3 * 21, embedding_dim, kernel_size=1)  

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_class)


        self.tp_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.tp_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.tp_fc3 = nn.Linear(hidden_dim // 2, num_class)

        # fuse feature projection for fpv and tpv
        self.fuse_feature_projection_fpv = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.fuse_feature_projection_tpv = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )



    def forward(self, visual_features, pose_features, fpv=True):
    
        B, _, T, J, _ = pose_features.size()
        pose_features = pose_features.view(B, -1, T)  # Shape (B, 3*J, T)
        pose_features = self.fpv_pose_projection(pose_features)  # Shape (B, in_channels, T)
        pose_features = pose_features.permute(0, 2, 1)  # Shape (B, T, in_channels)

        # Reshape visual features
        visual_features = self.visual_embedding(visual_features)
        visual_features = visual_features.permute(0, 2, 1) # Shape (B, T, C)
        initial_visual_features = visual_features

        # Cross-attention for visual features and pose features fusion
        visual_features = visual_features
        for layer in self.cross_attention_fp_layers:
            visual_features = layer(visual_features,pose_features, pose_features)
        fpv_feature = visual_features
        visual_features = initial_visual_features
        
        # Cross-attention for transformed pose features and pose features fusion into tpv
        for layer in self.cross_attention_tp_layers:
            visual_features = layer(visual_features, pose_features, pose_features)
        tpv_feature = visual_features

        if fpv:
            # Cross-attention for fused fpv and tpv features
            for layer in self.cross_attention_fp_fuse_layers:
                fused_features = layer(fpv_feature, tpv_feature, tpv_feature)

            feature = fused_features + tpv_feature + fpv_feature
            feature = self.fuse_feature_projection_fpv(fused_features)

            x = fused_features.mean(dim=1, keepdim=True)  # Temporal pooling to reduce T dimension
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)

            x = torch.sigmoid(x)
            x = x.squeeze(1)
            


        else:
            # Cross-attention for fused fpv and tpv features
            for layer in self.tp_cross_attention_fp_fuse_layers:
                fused_features = layer(fpv_feature, tpv_feature, tpv_feature)

            feature = fused_features + tpv_feature + fpv_feature
            feature = self.fuse_feature_projection_fpv(fused_features)

            x = fused_features.mean(dim=1, keepdim=True)  # Temporal pooling to reduce T dimension
            x = self.tp_fc1(x)
            x = self.relu(x)
            x = self.tp_fc2(x)
            x = self.relu(x)
            x = self.tp_fc3(x)

            x = torch.sigmoid(x)
            x = x.squeeze(1)
            
        return x, feature

