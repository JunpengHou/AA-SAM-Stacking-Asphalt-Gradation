from __future__ import annotations

import argparse
import os
from pathlib import Path

# Windows 环境中先导入 Pillow，避免 PyTorch/TorchVision 的 DLL 冲突。
from PIL import Image  # noqa: F401

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows 单机单 GPU 启动 Aggregate-SAM3，不初始化 Gloo/DDP。"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def configure_single_gpu_environment() -> None:
    # 仅满足 SAM3 日志模块对 rank 环境变量的读取，不初始化 torch.distributed。
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"


class SingleProcessDistributedSampler(TorchDistributedSampler):
    """
    单进程兼容采样器。

    官方 SAM3 TorchDataset 会无条件创建 DistributedSampler，但没有显式传入
    num_replicas/rank，PyTorch 因而调用 dist.get_world_size()/get_rank()。
    这里固定 num_replicas=1、rank=0，从而不需要初始化 Gloo/DDP。
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )


def install_sampler_patch() -> None:
    # 必须在 Hydra 实例化 Trainer 之前替换模块中的采样器符号。
    import sam3.train.data.torch_dataset as sam3_torch_dataset

    sam3_torch_dataset.DistributedSampler = SingleProcessDistributedSampler
    print(
        "[单卡采样器] 已安装：num_replicas=1，rank=0；"
        "未初始化 Gloo/DDP。"
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"训练配置文件不存在：{config_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用 CUDA GPU。")

    configure_single_gpu_environment()
    torch.cuda.set_device(0)

    # 数值稳定优先：关闭TF32近似计算，使用最高精度FP32矩阵乘法。
    # 这会略微降低速度，但可减少大型Transformer微调中的数值漂移。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print(
        "[数值精度] FP32最高精度；CUDA matmul TF32=False；"
        "cuDNN TF32=False。"
    )

    install_sampler_patch()

    print(f"[设备] CUDA:0；{torch.cuda.get_device_name(0)}")
    print(
        "[启动方式] Windows 单机单 GPU；"
        "不创建 Gloo 进程组，不使用网络端口。"
    )

    cfg = OmegaConf.load(config_path)
    print("################ Aggregate-SAM3 Config ################")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    main()
