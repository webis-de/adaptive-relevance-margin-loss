from lightning.pytorch.cli import LightningCLI, ArgsType

import torch
from src.util import LoggerSaveConfigCallback


def cli(args: ArgsType = None):
    # Make A100 go brrrr
    torch.set_float32_matmul_precision("medium")
    # Set up CLI
    _ = LightningCLI(
        args=args,
        save_config_kwargs={"overwrite": True},
        save_config_callback=LoggerSaveConfigCallback,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
