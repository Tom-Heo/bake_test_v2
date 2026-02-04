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
        self.config = config
        self.is_train = is_train

        # is_train 플래그에 따라 Train/Valid 경로 분기
        if is_train:
            self.root_dir = config.DIV2K_TRAIN_ROOT
        else:
            self.root_dir = config.DIV2K_VALID_ROOT

        # 이미지 파일 리스트 (.png만)
        self.image_files = [
            f for f in os.listdir(self.root_dir) if f.lower().endswith(".png")
        ]
        self.image_files.sort()

        # 색공간 변환기 (sRGB <-> Oklab)
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
        return torch.round(x * steps) / steps

    def _make_even_size(self, tensor):
        """
        4:2:0 서브샘플링을 위해 이미지 크기를 짝수로 맞춤 (Reflect Padding)
        """
        _, h, w = tensor.shape
        pad_h = 1 if (h % 2 != 0) else 0
        pad_w = 1 if (w % 2 != 0) else 0

        if pad_h + pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor

    def _chroma_subsampling(self, oklab):
        """
        Oklab 공간에서의 4:2:0 시뮬레이션
        """
        L, a, b = oklab.chunk(3, dim=0)

        # 1. Downsample (Nearest Neighbor)
        a_sub = F.interpolate(a.unsqueeze(0), scale_factor=0.5, mode="nearest")
        b_sub = F.interpolate(b.unsqueeze(0), scale_factor=0.5, mode="nearest")

        # 2. Upsample (Bilinear)
        h, w = L.shape[1], L.shape[2]
        a_deg = F.interpolate(a_sub, size=(h, w), mode="bilinear", align_corners=False)
        b_deg = F.interpolate(b_sub, size=(h, w), mode="bilinear", align_corners=False)

        return torch.cat([L, a_deg.squeeze(0), b_deg.squeeze(0)], dim=0)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        target_srgb = transforms.ToTensor()(img)

        # 2. Resize / Crop Strategy
        if self.is_train:
            # [수정됨] 1024x1024 Crop + 안전장치(pad_if_needed)
            # 이미지가 1024보다 작으면 Reflect Padding으로 채운 뒤 잘라냄 -> 에러 방지
            cropper = transforms.RandomCrop(
                512, pad_if_needed=True, padding_mode="reflect"
            )
            target_srgb = cropper(target_srgb)
        else:
            # 검증 시에는 원본 해상도 유지 (단, 짝수로 맞춤)
            target_srgb = self._make_even_size(target_srgb)

        # 3. Create Input (Degradation Process)
        # Step A: 6-bit Quantization
        bit_depth = getattr(self.config, "BIT_DEPTH_INPUT", 6)
        input_srgb = self._quantize(target_srgb, bit_depth)

        # Step B: RGB -> Oklab Conversion
        target_oklab = self.srgb_to_oklab(target_srgb.unsqueeze(0)).squeeze(0)
        input_oklab = self.srgb_to_oklab(input_srgb.unsqueeze(0)).squeeze(0)

        # Step C: Chroma Subsampling
        if getattr(self.config, "CHROMA_SUBSAMPLE", True):
            input_oklab = self._chroma_subsampling(input_oklab)

        return input_oklab, target_oklab
