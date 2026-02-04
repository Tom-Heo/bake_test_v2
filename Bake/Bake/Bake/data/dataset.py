import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# core/palette.py에서 모듈 로드
from core.palette import Palette


class DIV2KDataset(Dataset):
    def __init__(self, config, is_train=True):
        super().__init__()
        # config.DIV2K_ROOT는 train.py에서 Train/Valid 경로로 동적 할당됨
        self.root_dir = config.DIV2K_ROOT
        self.config = config
        self.is_train = is_train

        # 이미지 파일 리스트 (.png만)
        self.image_files = [
            f for f in os.listdir(self.root_dir) if f.lower().endswith(".png")
        ]
        self.image_files.sort()

        # 색공간 변환기 (sRGB <-> Oklab)
        # Palette 모듈은 Parameter가 없으므로 CPU에서 초기화해도 무방
        self.srgb_to_oklab = Palette.sRGBtoOklab()

    def __len__(self):
        return len(self.image_files)

    def _quantize(self, x, bits):
        """
        RGB 상태에서의 Bit-depth 다운샘플링 시뮬레이션
        Input: 0~1 Float Tensor
        Output: Quantized Tensor (Banding Noise 적용됨)
        """
        steps = (2**bits) - 1
        # 반올림(round)을 통해 계단 현상(Banding) 유발
        return torch.round(x * steps) / steps

    def _make_even_size(self, tensor):
        """
        4:2:0 서브샘플링을 위해 이미지 크기를 짝수로 맞춤 (Reflect Padding)
        홀수 픽셀이 남으면 다운샘플링 시 정보가 꼬일 수 있음.
        tensor: (C, H, W)
        """
        _, h, w = tensor.shape
        pad_h = 1 if (h % 2 != 0) else 0
        pad_w = 1 if (w % 2 != 0) else 0

        if pad_h + pad_w > 0:
            # (Left, Right, Top, Bottom)
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor

    def _chroma_subsampling(self, oklab):
        """
        Oklab 공간에서의 4:2:0 시뮬레이션
        Input: (3, H, W) Oklab Tensor
        """
        L, a, b = oklab.chunk(3, dim=0)  # (1, H, W)

        # 1. Downsample (Nearest Neighbor)
        # 색 정보를 절반으로 줄이면서 디테일 소실 유도
        a_sub = F.interpolate(a.unsqueeze(0), scale_factor=0.5, mode="nearest")
        b_sub = F.interpolate(b.unsqueeze(0), scale_factor=0.5, mode="nearest")

        # 2. Upsample (Bilinear)
        # 다시 원본 크기로 늘리면서 경계가 뭉개지는(Blur) 현상 시뮬레이션
        h, w = L.shape[1], L.shape[2]
        a_deg = F.interpolate(a_sub, size=(h, w), mode="bilinear", align_corners=False)
        b_deg = F.interpolate(b_sub, size=(h, w), mode="bilinear", align_corners=False)

        # 차원 복구 (1, 1, H, W) -> (1, H, W)
        return torch.cat([L, a_deg.squeeze(0), b_deg.squeeze(0)], dim=0)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # PIL 로드 -> Tensor 변환 (0~1 float, sRGB)
        img = Image.open(img_path).convert("RGB")
        target_srgb = transforms.ToTensor()(img)

        # 2. Even Size Padding (No Crop, Full Resolution)
        # 원본 해상도를 유지하되 짝수로 보정
        target_srgb = self._make_even_size(target_srgb)

        # 3. Create Input (Degradation Process)

        # Step A: 6-bit Quantization (RGB Domain)
        # 소스 영상 자체가 비트레이트가 낮은 상황 시뮬레이션
        input_srgb = self._quantize(target_srgb, self.config.BIT_DEPTH_INPUT)

        # Step B: RGB -> Oklab Conversion
        # Palette 모듈은 (B, C, H, W) 입력을 기대하므로 차원 확장/축소
        # Target(GT)와 Input(Degraded) 모두 Oklab으로 변환
        target_oklab = self.srgb_to_oklab(target_srgb.unsqueeze(0)).squeeze(0)
        input_oklab = self.srgb_to_oklab(input_srgb.unsqueeze(0)).squeeze(0)

        # Step C: Chroma Subsampling (Oklab Domain)
        # Oklab 상에서 색상(a, b) 채널만 뭉개버림
        if self.config.CHROMA_SUBSAMPLE:
            input_oklab = self._chroma_subsampling(input_oklab)

        return input_oklab, target_oklab
