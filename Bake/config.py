import os


class Config:
    # -------------------------------------------------------------------------
    # [Path Settings]
    # -------------------------------------------------------------------------
    # 데이터셋 저장/로드 루트
    DATA_DIR = "dataset"
    DIV2K_TRAIN_ROOT = os.path.join(DATA_DIR, "DIV2K_train_HR")
    DIV2K_VALID_ROOT = os.path.join(DATA_DIR, "DIV2K_valid_HR")

    # 체크포인트 및 로그 저장 경로
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULT_DIR = "results"

    # 마지막 학습 상태 자동 로드 파일명
    LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.pth")

    # -------------------------------------------------------------------------
    # [Data Settings]
    # -------------------------------------------------------------------------
    BATCH_SIZE = 1  # 이미지 크기가 다르므로 1 고정
    NUM_WORKERS = 4  # 데이터 로더 워커 수

    # 데이터 손상(Degradation) 파라미터
    BIT_DEPTH_INPUT = 3  # 3-bit Quantization
    CHROMA_SUBSAMPLE = True  # 4:2:0 Subsampling (Oklab a, b channel)

    # -------------------------------------------------------------------------
    # [Model Settings]
    # -------------------------------------------------------------------------
    BAKE_DIM = 96  # BakeNet 내부 채널 차원
    EMA_DECAY = 0.999  # EMA 감쇠율

    # -------------------------------------------------------------------------
    # [Training Settings]
    # -------------------------------------------------------------------------
    TOTAL_EPOCHS = 10000  # 총 학습 에폭

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6

    # Scheduler (ExponentialLR)
    SCHEDULER_GAMMA = 0.999996  # per step decay

    # 주기 설정
    LOG_INTERVAL_STEPS = 50  # 50 스텝마다 로그 출력
    VALID_INTERVAL_EPOCHS = 5  # 5 에폭마다 검증 및 체크포인트 저장

    # -------------------------------------------------------------------------
    # [Hardware Settings]
    # -------------------------------------------------------------------------
    DEVICE = "cuda"  # VRAM 96GB 사용
    USE_AMP = False  # AMP 미사용 (FP32 정밀도 유지)

    @classmethod
    def create_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.RESULT_DIR, exist_ok=True)
