import argparse
import yaml
import os
from utils.seed import set_seed
from utils.logging import Logger

def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return a dictionary."""
    with open(config_path, "r") as f:
        cfg=yaml.safe_load(f)
    return cfg


def create_dirs(cfg: dict):
    """Ensure logging and checkpoint directories exist"""

    log_dir = cfg["logging"]["log_dir"]
    ckpt_dir = cfg["logging"].get("checkpoint_dir", "outputs/checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    return log_dir, ckpt_dir


def main():
    """
    Experiment launcher for TinyGPT.

    Responsibilities:
    - Parse configuration from YAML
    - Set deterministic seed
    - Create logging & checkpoint directories
    - Initialize logger
    - Placeholder for dataset, model, optimizer, and training loop
    """
    
    # Load configuration from command line argument
    parser = argparse.ArgumentParser(description="Run TinyGPT experiment from config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[INFO] Loaded config from {args.config}")

    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"[INFO] Set deterministic seed: {seed}")

    # Create output directories
    log_dir, ckpt_dir = create_dirs(cfg)
    try:
        logger = Logger(log_dir)
        print(f"[INFO] Logging to {log_dir}, checkpoints to {ckpt_dir}")

        # TODO: 
        # - Load dataset according to cfg
        # - Initialize TinyGPT model using cfg parameters
        # - Initialize optimizer, scheduler
        # - Run training loop and log metrics

    finally:
        logger.close()
        print("[INFO] Experiment finished.")