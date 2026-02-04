from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


class Palette:
    class sRGBtoOklab(nn.Module):
        """
        Oklab 변환 (sRGB in [0,1] 가정).
        출력 채널: Lp, ap, bp (네가 사용 중인 스케일):
          Lp = 2*L - 1
          ap = 2*a
          bp = 2*b
        """

        def __init__(self):
            super().__init__()

        @staticmethod
        def srgb_to_lsrgb(srgb: torch.Tensor) -> torch.Tensor:
            return torch.where(
                srgb <= 0.04045,
                srgb / 12.92,
                ((srgb + 0.055) / 1.055) ** 2.4,
            )

        @staticmethod
        def lsrgb_to_oklab(
            lsred: torch.Tensor, lsgreen: torch.Tensor, lsblue: torch.Tensor
        ):
            def cbrt(x: torch.Tensor) -> torch.Tensor:
                return torch.sign(x) * torch.abs(x).pow(1.0 / 3.0)

            l = 0.4122214708 * lsred + 0.5363325363 * lsgreen + 0.0514459929 * lsblue
            m = 0.2119034982 * lsred + 0.6806995451 * lsgreen + 0.1073969566 * lsblue
            s = 0.0883024619 * lsred + 0.2817188376 * lsgreen + 0.6299787005 * lsblue

            l = cbrt(l)
            m = cbrt(m)
            s = cbrt(s)

            oklab_L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
            oklab_a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
            oklab_b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s

            # 범위 확장 (네 스케일)
            Lp = (2 * oklab_L) - 1.0
            ap = 2 * oklab_a
            bp = 2 * oklab_b

            return Lp, ap, bp

        def forward(self, x: torch.Tensor):
            # x: (B,3,H,W) assumed in [0,1]
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"Oklab expects (B,3,H,W), got shape={tuple(x.shape)}")

            x = x.permute(0, 2, 3, 1)  # NHWC
            srgb = x.clamp(0.0, 1.0)
            lsrgb = self.srgb_to_lsrgb(srgb)

            lsred = lsrgb[..., 0:1]
            lsgreen = lsrgb[..., 1:2]
            lsblue = lsrgb[..., 2:3]

            Lp, ap, bp = self.lsrgb_to_oklab(lsred, lsgreen, lsblue)
            nhwc = torch.cat([Lp, ap, bp], dim=-1)
            nchw = nhwc.permute(0, 3, 1, 2)
            return nchw

    class OklabtosRGB(nn.Module):
        """
        Oklab(Lp,ap,bp 스케일) -> sRGB [0,1]
        입력: (B,3,H,W)
          Lp∈[-1,1], ap,bp는 기존 Oklab 스케일의 2배
        출력: (B,3,H,W), sRGB [0,1]
        """

        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4 or x.size(1) != 3:
                raise ValueError(f"RGB expects (B,3,H,W), got shape={tuple(x.shape)}")

            # 스케일 복원: L = (Lp+1)/2, a = ap/2, b = bp/2
            Lp, ap, bp = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            L = (Lp + 1.0) * 0.5
            a = ap * 0.5
            b = bp * 0.5

            # Oklab -> LMS^3
            l_ = L + 0.3963377774 * a + 0.2158037573 * b
            m_ = L - 0.1055613458 * a - 0.0638541728 * b
            s_ = L - 0.0894841775 * a - 1.2914855480 * b

            l = l_**3
            m = m_**3
            s = s_**3

            # LMS -> linear sRGB
            r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
            g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
            b_rgb = 0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
            srgb_lin = torch.cat([r, g, b_rgb], dim=1)

            # linear sRGB -> sRGB 감마, [0,1]
            threshold = 0.0031308
            srgb = torch.where(
                srgb_lin <= threshold,
                12.92 * srgb_lin,
                1.055 * torch.clamp(srgb_lin, min=0.0) ** (1.0 / 2.4) - 0.055,
            )
            return srgb.clamp(0.0, 1.0)
