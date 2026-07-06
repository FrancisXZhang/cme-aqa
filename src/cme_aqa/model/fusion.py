import math

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.key_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.value_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, q, k, v):
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        query = self.query_conv(q)
        key = self.key_conv(k)
        value = self.value_conv(v)

        query = self.layer_norm1(query.permute(0, 2, 1))
        key = self.layer_norm1(key.permute(0, 2, 1))
        value = self.layer_norm1(value.permute(0, 2, 1))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attended = torch.matmul(self.softmax(scores), value)

        residual = v.permute(0, 2, 1)
        out = self.layer_norm2(attended + residual)
        return self.layer_norm2(self.ffn(out) + out)


class FusionModel(nn.Module):
    """CME-AQA CAT_PoseTrans_Dense_l4_EarlyShare-compatible model."""

    def __init__(self, in_channels=2048, num_class=3, hidden_dim=128, layers=4, ratio=8):
        super().__init__()
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
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_class)
        self.tp_fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.tp_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.tp_fc3 = nn.Linear(hidden_dim // 2, num_class)
        self.fuse_feature_projection_fpv = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fuse_feature_projection_tpv = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, visual_features, pose_features, fpv=True):
        batch, _, time_steps, joints, _ = pose_features.size()
        pose_features = pose_features.view(batch, -1, time_steps)
        pose_features = self.fpv_pose_projection(pose_features).permute(0, 2, 1)

        visual_features = self.visual_embedding(visual_features).permute(0, 2, 1)
        initial_visual = visual_features

        fp_layers = self.cross_attention_fp_layers if fpv else self.tp_cross_attention_fp_layers
        tp_layers = self.cross_attention_tp_layers if fpv else self.tp_cross_attention_tp_layers
        fuse_layers = self.cross_attention_fp_fuse_layers if fpv else self.tp_cross_attention_fp_fuse_layers

        for layer in fp_layers:
            visual_features = layer(visual_features, pose_features, pose_features)
        fpv_feature = visual_features

        visual_features = initial_visual
        for layer in tp_layers:
            visual_features = layer(visual_features, pose_features, pose_features)
        tpv_feature = visual_features

        fused_features = fpv_feature
        for layer in fuse_layers:
            fused_features = layer(fpv_feature, tpv_feature, tpv_feature)

        feature = self.fuse_feature_projection_fpv(fused_features)
        pooled = fused_features.mean(dim=1, keepdim=True)
        if fpv:
            x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(pooled)))))
        else:
            x = self.tp_fc3(self.relu(self.tp_fc2(self.relu(self.tp_fc1(pooled)))))
        return torch.sigmoid(x).squeeze(1), feature


Model = FusionModel


def build_fusion_model(variant="l4", in_channels=2048, num_class=3):
    """Build variants matching the historical CME-AQA model scripts."""
    if variant == "l2":
        return FusionModel(in_channels=in_channels, num_class=num_class, hidden_dim=256, layers=2, ratio=4)
    if variant == "l4":
        return FusionModel(in_channels=in_channels, num_class=num_class, hidden_dim=128, layers=4, ratio=8)
    raise ValueError(f"Unknown fusion model variant: {variant}")
