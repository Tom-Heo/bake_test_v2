import os
import sys
import logging
import torch
import shutil
import requests
from zipfile import ZipFile
from tqdm import tqdm


class ModelEMA:
    """
    Exponential Moving Average (EMA) for Model Parameters.
    학습 중 파라미터의 이동 평균을 유지하여 추론 시 더 안정적인 성능을 제공합니다.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}  # 원본 파라미터 백업용 딕셔너리

        # 초기화: 현재 모델의 파라미터를 그대로 복사
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """EMA 가중치를 모델에 덮어씌움 (검증 전 사용)"""
        # 먼저 원본을 백업하는 것이 안전하지만, 명시적 제어를 위해 store() 호출을 권장
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def store(self, model):
        """현재 모델의 파라미터를 백업합니다 (검증 전 호출)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def restore(self, model):
        """백업해둔 파라미터를 모델에 복구합니다 (검증 후 호출)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}  # 메모리 절약을 위해 비움


def get_logger(log_dir):
    """콘솔과 파일에 동시에 기록하는 로거 생성"""
    logger = logging.getLogger("BakeTrain")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_path = os.path.join(log_dir, "train.log")
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def download_file(url, save_path):
    """파일 다운로드 (Progress Bar 포함)"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(save_path, "wb") as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def prepare_div2k_dataset(config, logger):
    """DIV2K 데이터셋 확인 및 다운로드"""
    urls = {
        "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }

    if not os.path.exists(config.DIV2K_TRAIN_ROOT):
        logger.info("Dataset not found. Downloading DIV2K Train HR...")
        zip_path = os.path.join(config.DATA_DIR, "DIV2K_train_HR.zip")
        download_file(urls["train"], zip_path)

        logger.info("Extracting...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
        os.remove(zip_path)

    if not os.path.exists(config.DIV2K_VALID_ROOT):
        logger.info("Dataset not found. Downloading DIV2K Valid HR...")
        zip_path = os.path.join(config.DATA_DIR, "DIV2K_valid_HR.zip")
        download_file(urls["valid"], zip_path)

        logger.info("Extracting...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(config.DATA_DIR)
        os.remove(zip_path)

    logger.info(f"DIV2K dataset is ready at {config.DATA_DIR}")


def save_checkpoint(
    config, epoch, model, model_ema, optimizer, scheduler, is_best=False
):
    """체크포인트 저장"""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": model_ema.shadow,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    filename = os.path.join(config.CHECKPOINT_DIR, f"epoch_{epoch:05d}.pth")
    torch.save(state, filename)
    torch.save(state, config.LAST_CKPT_PATH)

    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
        shutil.copyfile(filename, best_path)


def load_checkpoint(config, model, model_ema, optimizer, scheduler, logger):
    """체크포인트 로드 (Resume)"""
    if not os.path.exists(config.LAST_CKPT_PATH):
        logger.info("No checkpoint found. Starting from scratch.")
        return 0

    logger.info(f"Loading checkpoint from {config.LAST_CKPT_PATH}")
    checkpoint = torch.load(config.LAST_CKPT_PATH, map_location=config.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "ema_shadow" in checkpoint:
        model_ema.shadow = checkpoint["ema_shadow"]
        logger.info("EMA state loaded.")

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    logger.info(f"Resuming training from epoch {start_epoch}")

    return start_epoch
