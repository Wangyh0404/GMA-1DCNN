# !/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch import nn
class GatedMultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(GatedMultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)


        self.gate_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        gated_output = self.gate_attention(attn_output)

        x = self.norm1(x + gated_output)  # 将门控与注意力模块结合

        return x



class GMA_1DCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10, embed_size=256, num_heads=1):
        super(GMA_1DCNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=256, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )


        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )


        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )


        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),  # 普通卷积
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)  # 输出形状: (batch_size, 256, 4)
        )


        self.attention_block = GatedMultiheadAttention(embed_size=embed_size, num_heads=num_heads)


        self.fc = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = x.transpose(1, 2)
        x = self.attention_block(x)


        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
