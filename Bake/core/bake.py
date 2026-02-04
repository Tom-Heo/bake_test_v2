from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
from .heo import Heo as hh
from .palette import Palette as pp


class Bake:
    class OklabtoBakedColor(nn.Module):
        def __init__(self):
            super().__init__()
            """
            Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            
            Macbeth ColorChecker 24 Reference Points (Float32 Precision)
            Lp, ap, bp 좌표계의 절대 기준점
            """
            MACBETH_REFS = [
                [-0.0597942, 0.0715682, 0.0689456],
                [0.4183730, 0.0844511, 0.0862138],
                [0.1485260, -0.0260145, -0.1197130],
                [0.0060500, -0.0867117, 0.1012940],
                [0.2456150, 0.0471529, -0.1390040],
                [0.4774960, -0.1772700, 0.0070673],
                [0.3507240, 0.1477280, 0.2429970],
                [-0.0020556, 0.0205867, -0.2380310],
                [0.1933670, 0.2535120, 0.0753901],
                [-0.1681210, 0.1244790, -0.1221760],
                [0.4962120, -0.1630230, 0.2584340],
                [0.5099230, 0.0561745, 0.2807750],
                [-0.1797940, 0.0282334, -0.2856320],
                [0.1974370, -0.2180970, 0.1560710],
                [0.0268793, 0.2907220, 0.1168620],
                [0.6649640, -0.0396422, 0.3292740],
                [0.1965930, 0.2874640, -0.0811799],
                [0.1415920, -0.1571350, -0.1312720],
                [0.9277470, -0.0007701, 0.0023781],
                [0.6655870, -0.0000168, -0.0001434],
                [0.4115140, -0.0000142, -0.0001216],
                [0.1583850, -0.0008645, 0.0027884],
                [-0.1009340, -0.0000091, -0.0000774],
                [-0.3499260, -0.0000066, -0.0000560],
            ]

            # 모델의 State로 등록 (Device 이동 자동화)
            self.register_buffer(
                "macbeth_refs",
                torch.tensor(MACBETH_REFS, dtype=torch.float32).view(1, 24, 3, 1, 1),
            )

            # 2. RBF Coefficients (Geometric Progression)
            # 0.1 (Micro) -> 0.4 (Meso) -> 1.6 (Macro)
            self.coeffs = [0.1, 0.4, 1.6]

        def forward(self, x):
            """
            Input:  x (B, 3, H, W) - Normalized Oklab (Lp, ap, bp)
            Lp = 2*L - 1
            ap = 2*a
            bp = 2*b
            Output: out (B, 96, H, W)
            """
            # --- Part 1: Signal Replication (24 Channels) ---
            # 원본 신호의 흐름을 강화하기 위해 8회 반복
            part1 = x.repeat(1, 8, 1, 1)

            # --- Part 2: RBF Global Context (72 Channels) ---
            # 1. Squared Euclidean Distance Calculation
            # (B, 1, 3, H, W) - (1, 24, 3, 1, 1) Broadcasting
            diff = x.unsqueeze(1) - self.macbeth_refs
            dist_sq = (diff**2).sum(dim=2)  # (B, 24, H, W)

            # 2. Gaussian RBF Application
            rbf_features = []
            for sigma in self.coeffs:
                # Gaussian Definition: exp( - d^2 / (2 * sigma^2) )
                gamma = 1.0 / (2.0 * (sigma**2))
                rbf = torch.exp(-dist_sq * gamma)
                rbf_features.append(rbf)

            part2 = torch.cat(rbf_features, dim=1)

            # --- Final Concatenation (96 Channels) ---
            baked_color = torch.cat([part1, part2], dim=1)

            return baked_color

    class BakedColortoOklab(nn.Module):
        """
        96채널 BakedColor에서 순수 신호(Lp, ap, bp)를 추출하여 평균을 냄.
        Bake.OklabtoBakedColor의 역연산 개념.
        """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 96, H, W)

            # Part 1: Signal Replication 영역 (0~23번 채널) 추출
            # 구조: [L, a, b, L, a, b, ...] 형태가 8번 반복됨
            signal_part = x[:, :24, :, :]

            # (B, 8, 3, H, W)로 Reshape하여 8번의 반복을 차원으로 분리
            # dim 1이 반복 횟수(8), dim 2가 채널(3: L,a,b)
            reshaped = signal_part.view(x.size(0), 8, 3, x.size(2), x.size(3))

            # 8개의 반복된 신호에 대해 평균을 구함 -> 노이즈 캔슬링 효과
            avg_signal = reshaped.mean(dim=1)  # (B, 3, H, W)

            return avg_signal

    class BakedColortoColorEmbedding(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = nn.Conv2d(96, 96, 1, 1)
            self.conv2 = nn.Conv2d(96, 96, 1, 1)
            self.conv3 = nn.Conv2d(96, 96, 1, 1)
            self.act1 = hh.HeLU2d(96)
            self.act2 = hh.HeLU2d(96)
            self.act3 = hh.HeLU2d(96)
            self.gate1 = hh.HeoGate2d(96)
            self.gate2 = hh.HeoGate2d(96)
            self.gate3 = hh.HeoGate2d(96)

        def forward(self, baked_color):
            """
            Input: baked_color (B, 96, H, W)
            """

            x1 = self.conv1(baked_color)
            x1 = self.act1(x1)
            x1 = self.gate1(baked_color, x1)

            x2 = self.conv2(x1)
            x2 = self.act2(x2)
            x2 = self.gate2(x1, x2)

            x3 = self.conv3(x2)
            x3 = self.act3(x3)
            color_embedding = self.gate3(x2, x3)

            return color_embedding

    class BakeLoss(nn.Module):
        def __init__(self, epsilon=0.001):
            super().__init__()
            self.epsilon = epsilon
            self.epsilon_char = 1e-8

        def forward(self, pred, target):
            pred = pred.float()  # FP32 강제
            target = target.float()

            diff = pred - target
            charbonnier = torch.sqrt(diff**2 + self.epsilon_char**2)

            loss = torch.log(1 + charbonnier / self.epsilon)

            return loss.mean()

    class BakeBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.color_embedding_encoder = Bake.BakedColortoColorEmbedding()

            self.color_embedding_gate = hh.HeoGate2d(dim)
            self.residual_gate = hh.HeoGate2d(dim)

            self.nemo1 = hh.NeMO33(dim)
            self.nemo2 = hh.NeMO33(dim)

        def forward(self, x, baked_x_lr):
            color_embedding = self.color_embedding_encoder(baked_x_lr)
            residual = x

            x = self.nemo1(x)
            x = self.color_embedding_gate(x, color_embedding)
            x = self.nemo2(x)
            x = self.residual_gate(x, residual)

            return x

    class BakeNet(nn.Module):
        def __init__(self, dim=96):
            super().__init__()

            self.block1 = Bake.BakeBlock(dim)
            self.block2 = Bake.BakeBlock(dim)
            self.block3 = Bake.BakeBlock(dim)
            self.block4 = Bake.BakeBlock(dim)
            self.block5 = Bake.BakeBlock(dim)
            self.block6 = Bake.BakeBlock(dim)
            self.block7 = Bake.BakeBlock(dim)
            self.block8 = Bake.BakeBlock(dim)

        def forward(self, baked_x_lr):

            x = self.block1(baked_x_lr, baked_x_lr)
            x = self.block2(x, baked_x_lr)
            x = self.block3(x, baked_x_lr)
            x = self.block4(x, baked_x_lr)
            x = self.block5(x, baked_x_lr)
            x = self.block6(x, baked_x_lr)
            x = self.block7(x, baked_x_lr)
            x = self.block8(x, baked_x_lr)

            return x

    class BakeModel(nn.Module):
        def __init__(self, dim=96):
            super().__init__()
            self.encoder = Bake.OklabtoBakedColor()
            self.body = Bake.BakeNet(dim)
            self.decoder = Bake.BakedColortoOklab(dim)

        def forward(self, x):

            baked_x = self.encoder(x)
            baked_residual = self.body(baked_x)
            out = baked_x + (baked_residual * 0.2)
            out = self.decoder(out)

            return out
