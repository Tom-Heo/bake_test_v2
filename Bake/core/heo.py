from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


class Heo:
    class HeLU(nn.Module):
        """
        원본 HeLU: last-dim 기반 (..., dim) 입력용
        """

        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))
            self.redweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(dim).normal_(mean=0.0, std=0.45))
            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor):
            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            red = torch.tanh(sqrt(3.0) * self.redweight) + 1.0
            blue = torch.tanh(sqrt(3.0) * self.blueweight) + 1.0
            redx = rgx * red
            bluex = bgx * blue
            x = redx + bluex
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            y = (alpha * x + beta * raw) / 2
            return y

    class HeLU2d(nn.Module):
        """
        입력: (N,C,H,W)
        """

        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            # 원본 HeLU와 같은 파라미터 의미(채널별)
            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))
            self.redweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))
            self.blueweight = nn.Parameter(torch.empty(c).normal_(mean=0.0, std=0.45))

            self.redgelu = nn.GELU()
            self.bluegelu = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise ValueError(
                    f"HeLU2d expects NCHW 4D tensor, got shape={tuple(x.shape)}"
                )
            if x.size(1) != self.channels:
                raise ValueError(
                    f"HeLU2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            raw = x

            rgx = self.redgelu(x)
            bgx = -1.0 * self.bluegelu(-x)

            # (C,) -> (1,C,1,1) broadcasting
            red = (torch.tanh(sqrt(3.0) * self.redweight) + 1.0).view(1, -1, 1, 1)
            blue = (torch.tanh(sqrt(3.0) * self.blueweight) + 1.0).view(1, -1, 1, 1)
            x = rgx * red + bgx * blue

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            y = (alpha * x + beta * raw) / 2
            return y

    class HeoGate(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.alpha = nn.Parameter(torch.full((dim,), 0.9))
            self.beta = nn.Parameter(torch.full((dim,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            alpha = torch.tanh(sqrt(3.0) * self.alpha) + 1.0
            beta = torch.tanh(sqrt(3.0) * self.beta) + 1.0
            return (alpha * x + beta * raw) / 2

    class HeoGate2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.channels = c

            self.alpha = nn.Parameter(torch.full((c,), 0.9))
            self.beta = nn.Parameter(torch.full((c,), -0.9))

        def forward(self, x: torch.Tensor, raw: torch.Tensor):
            if x.dim() != 4 or x.size(1) != self.channels:
                raise ValueError(
                    f"HeoGate2d channels mismatch: expected C={self.channels}, got C={x.size(1)}"
                )

            alpha = (torch.tanh(sqrt(3.0) * self.alpha) + 1.0).view(1, -1, 1, 1)
            beta = (torch.tanh(sqrt(3.0) * self.beta) + 1.0).view(1, -1, 1, 1)
            return (alpha * x + beta * raw) / 2

    class HeoTimeEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                Heo.HeLU(dim * 4),
                nn.Linear(dim * 4, dim * 4),
                Heo.HeLU(dim * 4),
                nn.Linear(dim * 4, dim),
            )

        def forward(self, time):
            # Sinusoidal embedding
            device = time.device
            half_dim = self.dim // 2
            embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return self.mlp(embeddings)

    class HeoTimeInjection2d(nn.Module):
        """
        HeoTimeInjection2d: Adaptive Time Injection Module
        스케일(Scale)과 시프트(Shift)를 조절합니다.
        """

        def __init__(self, dim):
            super().__init__()

            # 2. Time Embedding으로부터 Scale과 Shift를 예측하는 Act + Linear
            self.act1 = Heo.HeLU(dim)
            self.act2 = Heo.HeLU(dim * 2)
            self.linear1 = nn.Linear(dim, dim * 2)  # Output: (Scale, Shift)
            self.linear2 = nn.Linear(dim * 2, dim * 2)  # Output: (Scale, Shift)
            # 3. Zero Initialization
            # 학습 초기에는 입력(x)을 그대로 내보내도록 Scale/Shift를 0에 가깝게 초기화
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)

        def forward(self, x, t_emb):
            """
            x: (B, C, H, W)
            t_emb: (B, C) - Time Embedding
            """
            # A. 시간 정보로부터 변조 계수 예측
            emb = self.linear1(self.act1(t_emb))
            emb = self.linear2(self.act2(emb))
            scale, shift = emb.chunk(2, dim=1)  # (B, C), (B, C)

            # Broadcasting을 위한 차원 확장 (B, C, 1, 1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]

            # B. Adaptive Modulation
            x = x * (torch.tanh(sqrt(3.0) * scale) + 1.0) + shift
            return x

    class NeMO33(nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.conv1 = nn.Conv2d(dim, dim, 1, 1)
            self.conv2 = nn.Conv2d(dim, dim, 1, 1)
            self.conv3 = nn.Conv2d(dim, dim, 1, 1)

            self.act0 = Heo.HeLU2d(dim)
            self.act1 = Heo.HeLU2d(dim)
            self.act2 = Heo.HeLU2d(dim)
            self.act3 = Heo.HeLU2d(dim)

            self.gate0 = Heo.HeoGate2d(dim)
            self.gate1 = Heo.HeoGate2d(dim)
            self.gate2 = Heo.HeoGate2d(dim)
            self.gate3 = Heo.HeoGate2d(dim)

        def forward(self, x: torch.Tensor):

            raw = x
            x = self.conv0(x)
            x = self.act0(x)
            x = self.gate0(x, raw)

            x0 = x
            x = self.conv1(x)
            x = self.act1(x)
            x = self.gate1(x0, x)

            x1 = x
            x = self.conv2(x)
            x = self.act2(x)
            x = self.gate2(x1, x)

            x2 = x
            x = self.conv3(x)
            x = self.act3(x)
            x = self.gate3(x2, x)

            return x
