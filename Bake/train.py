import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Local Modules
from config import Config
from utils import (
    get_logger,
    prepare_div2k_dataset,
    save_checkpoint,
    load_checkpoint,
    ModelEMA,
)
from data.dataset import DIV2KDataset
from core.bake import Bake
from core.palette import Palette


def compute_psnr(img1, img2):
    """
    PSNR 계산 (RGB 0~1 range)
    img1, img2: (B, 3, H, W) Tensor
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * torch.log10(1.0 / mse)


def train(config, args):
    # 1. Setup
    Config.create_directories()
    logger = get_logger(config.LOG_DIR)
    device = torch.device(config.DEVICE)

    logger.info(f"Starting Bake Training on {device}")

    # 데이터셋 준비 (다운로드 확인)
    prepare_div2k_dataset(config, logger)

    # 2. Data Loaders
    logger.info("Initializing Datasets...")
    train_dataset = DIV2KDataset(config, is_train=True)
    valid_dataset = DIV2KDataset(config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,  # config에서 설정한 값 (기본 1)
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,  # 검증도 1장씩 (해상도 다름)
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Components Initialization
    logger.info("Initializing Models...")

    # 학습 대상 모델
    model = Bake.BakeNet(dim=config.BAKE_DIM).to(device)

    # 변환기 (Freeze) - 학습 파라미터 없음
    # 3ch -> 96ch
    to_baked = Bake.OklabtoBakedColor().to(device)
    # 96ch -> 3ch (검증용)
    to_oklab = Bake.BakedColortoOklab().to(device)
    # 3ch -> RGB (시각화용)
    to_rgb = Palette.OklabtosRGB().to(device)

    # 손실 함수
    criterion = Bake.BakeLoss().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Per-step Scheduler이므로 step()을 매 배치마다 호출
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.SCHEDULER_GAMMA
    )

    # EMA
    model_ema = ModelEMA(model, decay=config.EMA_DECAY)

    # 4. Resume Logic (Option Based)
    if args.restart:
        logger.info(
            "[Option] --restart detected. Ignoring checkpoints, starting from scratch."
        )
        start_epoch = 0
    else:
        # --resume이거나 옵션이 없을 때는 기본적으로 Load 시도
        # (체크포인트가 없으면 내부적으로 0을 리턴하므로 안전)
        if args.resume:
            logger.info("[Option] --resume detected. Attempting to load checkpoint.")
        else:
            logger.info("No option specified. Defaulting to resume logic.")

        start_epoch = load_checkpoint(
            config, model, model_ema, optimizer, scheduler, logger
        )

    # 5. Training Loop
    logger.info(f"Start Training Loop from Epoch {start_epoch}")

    for epoch in range(start_epoch, config.TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for step, (input_3ch, target_3ch) in enumerate(train_loader, 1):
            # Data to GPU: (B, 3, H, W) Oklab
            input_3ch = input_3ch.to(device)
            target_3ch = target_3ch.to(device)

            # --- [Core Logic: 3ch -> 96ch Conversion] ---
            # 학습 효율을 위해 GPU에서 변환 수행
            with torch.no_grad():
                input_96ch = to_baked(input_3ch)  # (B, 96, H, W)
                target_96ch = to_baked(target_3ch)  # (B, 96, H, W)

            # --- Forward ---
            # Input도 96ch로 변환된 상태로 BakeNet에 진입
            pred_96ch = model(input_96ch)

            # --- Loss (96ch Space) ---
            loss = criterion(pred_96ch, target_96ch)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()

            # (No Gradient Clipping as requested)
            optimizer.step()

            # --- Update EMA & Scheduler ---
            model_ema.update(model)
            scheduler.step()

            epoch_loss += loss.item()

            # Logging
            if step % config.LOG_INTERVAL_STEPS == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch [{epoch}/{config.TOTAL_EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f} "
                    f"LR: {current_lr:.2e}"
                )

        # Epoch Loss Summary
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"==> Epoch {epoch} Finished. Avg Loss: {avg_loss:.6f}")

        # 6. Validation Loop (Every 10 Epochs)
        # 0 Epoch(시작 직후 상태)도 확인하고 싶다면 epoch >= 0 조건 추가 가능하지만
        # 보통은 학습 후 확인하므로 interval 체크만 유지
        if epoch % config.VALID_INTERVAL_EPOCHS == 0:
            logger.info(f"==> Validating at Epoch {epoch}...")

            # A. EMA 적용 (원본 백업)
            model_ema.store(model)
            model_ema.apply_shadow(model)

            model.eval()
            val_psnr_accum = 0.0
            num_val_imgs = 0

            # 시각화용 이미지 홀더
            vis_save_path = os.path.join(
                config.RESULT_DIR, f"valid_epoch_{epoch:05d}.png"
            )
            saved_vis = False

            with torch.no_grad():
                for v_step, (v_in_3ch, v_gt_3ch) in enumerate(valid_loader):
                    v_in_3ch = v_in_3ch.to(device)
                    v_gt_3ch = v_gt_3ch.to(device)

                    # 1. Convert Input 3ch -> 96ch
                    v_in_96ch = to_baked(v_in_3ch)

                    # 2. Inference (Network Output: 96ch)
                    v_pred_96ch = model(v_in_96ch)

                    # 3. Restore to RGB for Metric & Visualization
                    # (96ch -> 3ch Oklab)
                    v_pred_3ch = to_oklab(v_pred_96ch)

                    # (3ch Oklab -> 3ch RGB)
                    v_pred_rgb = to_rgb(v_pred_3ch)
                    v_gt_rgb = to_rgb(v_gt_3ch)
                    v_in_rgb = to_rgb(v_in_3ch)  # Input 상태 확인용

                    # 4. Compute PSNR (RGB Domain)
                    psnr = compute_psnr(v_pred_rgb, v_gt_rgb)
                    val_psnr_accum += psnr.item()
                    num_val_imgs += 1

                    # 5. Save Visualization (First batch only)
                    if not saved_vis:
                        # (Input | Pred | Target) 가로로 병합
                        combined = torch.cat([v_in_rgb, v_pred_rgb, v_gt_rgb], dim=3)
                        save_image(combined, vis_save_path)
                        saved_vis = True

            avg_psnr = val_psnr_accum / num_val_imgs
            logger.info(f"==> Validation PSNR: {avg_psnr:.2f} dB")

            # B. EMA 해제 (원본 복구)
            model_ema.restore(model)

            # Save Checkpoint
            save_checkpoint(
                config,
                epoch,
                model,
                model_ema,
                optimizer,
                scheduler,
                is_best=False,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake Training Script")

    # 상호 배타적 그룹 생성 (동시에 사용 불가)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint (default behavior).",
    )
    group.add_argument(
        "--restart",
        action="store_true",
        help="Force restart training from scratch (Epoch 0), ignoring checkpoints.",
    )

    args = parser.parse_args()

    train(Config, args)
