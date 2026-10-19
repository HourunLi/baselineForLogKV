# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Extract a trained DMS student model from an intermediate checkpoint.

Usage:
    python -m models.qwen3.extract \
        --config outputs/qwen3_1.7b_base/config.yaml \
        --checkpoint outputs/qwen3_1.7b_base/checkpoint-238 \
        --output outputs/qwen3_1.7b_base/student_model_step238
"""

# Modified for the SemanticLogKV baseline: checkpoints of dms.training.loop hold the trained adapter weights;
# the rest of the student is the verified prepared Base.
import argparse
from pathlib import Path

import torch
from dms.logging import get_logger
from dms.training.engine import DistillationModelArguments, DMSTrainerState, ModelArguments
from dms.training.loop import TrainerSettings
from transformers import AutoTokenizer

from .prepare_base import verify_prepared_base
from .train import build_combined_model, extract_student_model, load_config

logger = get_logger("Extract")


def main() -> None:
    """Extract student model from a training checkpoint."""
    parser = argparse.ArgumentParser(
        description="Extract DMS student model from a training checkpoint."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training YAML config"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (defaults to checkpoint/student_model)",
    )
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)
    model_cfg = cfg["model"]
    dms_cfg = cfg["dms"]
    data_cfg = cfg["data"]

    checkpoint_dir = Path(cli_args.checkpoint)
    adapters_path = checkpoint_dir / "adapters.pt"
    save_path = cli_args.output or str(checkpoint_dir / "student_model")

    logger.info(f"Loading adapters from: {adapters_path}")
    logger.info(f"Saving model to: {save_path}")

    verify_prepared_base(model_cfg["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_args = DistillationModelArguments(
        student=ModelArguments(
            model_name_or_path=model_cfg["name"],
            dtype=model_cfg.get("dtype", "float32"),
        ),
        teacher=ModelArguments(
            model_name_or_path=model_cfg.get("teacher_name", model_cfg["name"]),
            dtype=model_cfg.get("teacher_dtype", model_cfg.get("dtype", "float32")),
        ),
    )

    training_args = TrainerSettings.from_config(cfg["trainer"])
    trainer_state = DMSTrainerState()

    logger.info("Creating combined model...")
    combined_model = build_combined_model(
        model_args=model_args,
        training_args=training_args,
        dms_cfg=dms_cfg,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        trainer_state=trainer_state,
    )

    logger.info("Loading checkpoint weights...")
    adapters = torch.load(adapters_path, map_location="cpu", weights_only=True)
    trainable = sorted(name for name, param in combined_model.named_parameters() if param.requires_grad)
    if sorted(adapters) != trainable:
        raise ValueError(f"{adapters_path} does not hold exactly this model's trainable weights")
    combined_model.load_state_dict(adapters, strict=False)

    extract_student_model(combined_model, tokenizer, save_path)


if __name__ == "__main__":
    main()
