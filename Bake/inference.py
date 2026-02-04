import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Local Modules
from config import Config
from core.bake import Bake
from core.palette import Palette


def load_image(path):
    """이미지 로드 및 텐서 변환 (RGB, 0~1)"""
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img)


def pad_image(tensor):
    """
    홀수 해상도일 경우 짝수로 패딩 (Reflect)
    Returns: padded_tensor, (original_h, original_w)
    """
    _, h, w = tensor.shape
    pad_h = 1 if (h % 2 != 0) else 0
    pad_w = 1 if (w % 2 != 0) else 0

    if pad_h + pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, (h, w)


def unpad_image(tensor, original_size):
    """
    패딩된 텐서를 원래 크기로 복구 (Input: B, C, H, W)
    """
    h, w = original_size
    # 4차원 텐서 슬라이싱
    return tensor[:, :, :h, :w]


def inference(args):
    # 1. Setup
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Model Initialization
    print("Initializing Model...")
    model = Bake.BakeNet(dim=Config.BAKE_DIM).to(device)

    # Converters
    to_baked = Bake.OklabtoBakedColor().to(device)
    to_oklab = Bake.BakedColortoOklab().to(device)
    srgb_to_oklab = Palette.sRGBtoOklab().to(device)  # Input 전처리용
    oklab_to_srgb = Palette.OklabtosRGB().to(device)  # Output 후처리용

    # 3. Load Checkpoint (EMA)
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load EMA Weights if available (Preferred)
    if "ema_shadow" in checkpoint:
        print("Loading EMA weights (ema_shadow)...")
        model.load_state_dict(checkpoint["ema_shadow"])
    else:
        print("Warning: EMA weights not found. Loading standard model weights.")
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # 4. Input Processing
    if os.path.isdir(args.input):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        image_paths = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith(exts)
        ]
        save_dir = os.path.join(Config.RESULT_DIR, "inference_folder")
    else:
        image_paths = [args.input]
        save_dir = os.path.join(Config.RESULT_DIR, "inference_single")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(image_paths)} images. Saving results to {save_dir}")

    # 5. Inference Loop
    for img_path in image_paths:
        img_name = os.path.basename(img_path)

        # A. Load & Preprocess
        input_srgb = load_image(img_path).to(device)  # (3, H, W)

        # B. Padding (짝수 맞춤)
        input_padded, org_size = pad_image(input_srgb)

        # Batch 차원 추가 (1, 3, H, W)
        input_padded = input_padded.unsqueeze(0)

        with torch.no_grad():
            # C. RGB -> Oklab (3ch)
            input_oklab = srgb_to_oklab(input_padded)

            # D. Oklab -> Baked (96ch)
            input_baked = to_baked(input_oklab)

            # E. Network Inference
            output_baked = model(input_baked)

            # F. Baked -> Oklab (3ch)
            output_oklab = to_oklab(output_baked)

            # G. Oklab -> RGB
            output_rgb = oklab_to_srgb(output_oklab)

        # H. Unpad & Clamp
        output_rgb = unpad_image(output_rgb, org_size)  # (1, 3, H, W)
        output_rgb = output_rgb.squeeze(0)  # (3, H, W)
        output_rgb = output_rgb.clamp(0, 1)

        # I. Save Results (Two Versions)

        # 1) Comparison [Input | Output]
        combined = torch.cat([input_srgb, output_rgb], dim=2)
        save_path_comp = os.path.join(save_dir, f"comp_{img_name}")
        save_image(combined, save_path_comp)

        # 2) Result Only [Output]
        save_path_res = os.path.join(save_dir, f"res_{img_name}")
        save_image(output_rgb, save_path_res)

        print(f"Saved: {save_path_comp} & {save_path_res}")

    print("Inference Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake Inference")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a directory containing images.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=Config.LAST_CKPT_PATH,
        help="Path to the checkpoint file (.pth). Default: checkpoints/last.pth",
    )

    args = parser.parse_args()
    inference(args)
