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

# Modified for the SemanticLogKV baseline: Base text data, safe resume, distributed export, and
# dms.training.loop instead of HF Trainer/accelerate so training runs in SemanticLogKV's torch218.
"""Training entry point for Qwen3 model with DMS.

Usage:
    # First run prepares the dataset (single process):
    python -m models.qwen3.train --config configs/qwen3_1.7b_base.yaml --prepare-dataset-only

    # Then launch training (one or more processes):
    torchrun --standalone --nproc_per_node 8 -m models.qwen3.train --config configs/qwen3_1.7b_base.yaml
"""

import argparse
import functools
import json
import os
import shutil
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from dms.logging import get_logger
from dms.training.data import ConfiguredTokenizer, DataBlend, DataBlendElement, DatasetInfo
from dms.training.engine import (
    CombinedModel,
    DistillationModelArguments,
    DMSTrainerState,
    ModelArguments,
    distillation_forward,
    dms_schedule,
    get_student_model,
    get_teacher_model,
    get_tokenizer,
)
from dms.training.loop import DMSTrainer, TrainerSettings, checkpoints, init_runtime
from transformers import PreTrainedTokenizer, Qwen3ForCausalLM

from .modeling_qwen3_dms import Qwen3ForCausalLMDMS

logger = get_logger("Train")


# =============================================================================
# Config loading
# =============================================================================


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, output_dir: str) -> None:
    """Save the configuration to the output directory for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {config_path}")


def resolve_checkpoint(cfg: dict) -> str | None:
    """Resolve the checkpoint path for resume, supporting 'auto' detection."""
    resume = cfg["trainer"].get("resume_from_checkpoint")
    if resume == "auto":
        found = checkpoints(Path(cfg["trainer"]["output_dir"]))
        if found:
            logger.info(f"Auto-detected latest checkpoint: {found[-1]}")
            return str(found[-1])
        else:
            logger.info("No checkpoints found, starting fresh.")
            return None
    if resume is not None and not Path(resume).is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
    return resume


# =============================================================================
# Dataset parsing
# =============================================================================


def _parse_data_blend_elements(blend_string: str) -> list[DataBlendElement]:
    """Parse data blend elements from a comma-separated string.

    Args:
        blend_string: Comma-separated string of "dataset_name:weight" pairs.

    Returns:
        List of DataBlendElement objects.
    """
    import dms.training.data as rf_datasets

    elements = []
    for entry in blend_string.split(","):
        dataset_name, weight_str = entry.split(":")
        dataset = getattr(rf_datasets, dataset_name)
        weight = float(weight_str)
        logger.info(f"Adding dataset {dataset_name} with weight {weight}")
        elements.append(DataBlendElement(dataset=dataset, weight=weight))
    return elements


def _keep_text_sample(sample: dict) -> bool:
    return True


def _extract_text_sample(sample: dict) -> dict:
    text = sample["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each local JSONL record must contain a nonempty string 'text'")
    return {"conversation": [{"role": "user", "content": text}]}


def _create_train_dataset(
    data_cfg: dict,
    configured_tokenizer: ConfiguredTokenizer,
) -> Dataset:
    """Create the training dataset from data blend configuration."""
    if bool(data_cfg.get("train_file")) == bool(data_cfg.get("blend")):
        raise ValueError("Set exactly one of data.train_file or data.blend")
    if data_cfg.get("train_file"):
        train_file = Path(data_cfg["train_file"]).resolve(strict=True)
        local_data = DatasetInfo(
            args=("json",), kwargs={"data_files": str(train_file), "split": "train"},
            filter_fn=_keep_text_sample, extract_fn=_extract_text_sample,
            # Local data may change at the same path; rely on HF's file fingerprint.
            caching_enabled=False,
        )
        data_blend_elements = [DataBlendElement(dataset=local_data, weight=1.0)]
    else:
        data_blend_elements = _parse_data_blend_elements(data_cfg["blend"])

    data_blend = DataBlend(
        data_blend_elements=data_blend_elements,
        configured_tokenizer=configured_tokenizer,
        train_samples=data_cfg["train_samples"],
        concat_up_to=data_cfg["max_length"],
        concat_always_start_new=data_cfg.get("concat_always_start_new", True),
    )

    return Dataset.from_generator(lambda: (data_blend[i] for i in range(len(data_blend))))


# =============================================================================
# Model building
# =============================================================================


def build_combined_model(
    model_args: DistillationModelArguments,
    training_args: TrainerSettings,
    dms_cfg: dict,
    data_cfg: dict,
    tokenizer: PreTrainedTokenizer,
    trainer_state: DMSTrainerState,
) -> CombinedModel:
    """Build the combined student-teacher model for distillation."""
    dms_kwargs = {
        f"dms_{k}" if not k.startswith("dms_") else k: v
        for k, v in dms_cfg.items()
        if k not in ("initial_cr", "final_cr", "final_step")
    }

    student_model = get_student_model(
        model_args,
        zero_out_proj_alpha=True,
        model_constructor=Qwen3ForCausalLMDMS,
        dms_kwargs=dms_kwargs,
    )
    # This is a training target recorded as metadata, not an inference budget knob.
    student_model.config.dms_cr = dms_cfg["final_cr"]

    student_is_teacher = (
        model_args.student.model_name_or_path == model_args.teacher.model_name_or_path
        and model_args.student.dtype == model_args.teacher.dtype
    )
    if student_is_teacher:
        logger.info("Student and teacher are the same model - optimization enabled")
        teacher_model = student_model
    else:
        logger.info("Student and teacher are different models")
        teacher_model = get_teacher_model(model_args, model_constructor=Qwen3ForCausalLM)

    return CombinedModel(
        student_model=student_model,
        teacher_model=teacher_model,
        trainer_state=trainer_state,
        dms_schedule=functools.partial(
            dms_schedule,
            training_args=training_args,
            dms_initial_cr=dms_cfg["initial_cr"],
            dms_final_cr=dms_cfg["final_cr"],
            dms_final_step=dms_cfg.get("final_step"),
        ),
        forward_fn=distillation_forward,
        student_is_teacher=student_is_teacher,
        tokenizer=tokenizer,
        process_vocab_using_chunk=data_cfg.get("process_vocab_using_chunk", 4096),
        forward_fn_kwargs_student=model_args.student.forward_fn_kwargs,
        forward_fn_kwargs_teacher=model_args.teacher.forward_fn_kwargs,
    )


# =============================================================================
# Student model extraction
# =============================================================================

AUTO_MAP_CONFIG = {
    "AutoConfig": "configuration_qwen3_dms.Qwen3ConfigDMS",
    "AutoModel": "modeling_qwen3_dms.Qwen3ModelDMS",
    "AutoModelForCausalLM": "modeling_qwen3_dms.Qwen3ForCausalLMDMS",
    "AutoModelForQuestionAnswering": "modeling_qwen3_dms.Qwen3ForQuestionAnsweringDMS",
    "AutoModelForSequenceClassification": "modeling_qwen3_dms.Qwen3ForSequenceClassificationDMS",
    "AutoModelForTokenClassification": "modeling_qwen3_dms.Qwen3ForTokenClassificationDMS",
}


def extract_student_model(
    combined_model: CombinedModel,
    tokenizer: PreTrainedTokenizer,
    save_path: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> None:
    """Extract the student model from a CombinedModel and save it for inference.

    The saved model includes:
    - Model weights in bfloat16
    - Config with auto_map for trust_remote_code
    - Model implementation files (config.py, model.py)
    - Tokenizer

    Note: The saved model imports from the `dms` package. Make sure `dms` is
    installed (pip install -e .) in any environment where you load this model.
    """
    student_model = combined_model.student_model
    logger.info(f"Extracting student model to: {save_path}")

    if state_dict is None:
        state_dict = student_model.state_dict()
    state_dict = {name: value.detach().to(device="cpu", dtype=torch.bfloat16)
                  if value.is_floating_point() else value.detach().cpu()
                  for name, value in state_dict.items()}
    # Never cast live FSDP shards in-place during export.
    student_model.save_pretrained(save_path, state_dict=state_dict)
    tokenizer.save_pretrained(save_path)

    # Update config.json with auto_map
    config_path = Path(save_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config.pop("architectures", None)
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    config["auto_map"] = AUTO_MAP_CONFIG
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Copy model implementation files for trust_remote_code
    model_dir = Path(__file__).parent
    for src_name in ["configuration_qwen3_dms.py", "modeling_qwen3_dms.py"]:
        shutil.copy(model_dir / src_name, Path(save_path) / src_name)

    logger.info(f"Successfully saved student model to: {save_path}")


def export_trained_student(trainer, combined_model, tokenizer, save_path, training_record=None):
    """Gather FSDP weights on all ranks, then write one complete student on rank zero."""
    trainer.runtime.barrier()
    full_state = trainer.full_state_dict()
    if trainer.runtime.is_main:
        prefix = "student_model."
        student_state = {key[len(prefix):]: value for key, value in full_state.items()
                         if key.startswith(prefix)}
        if not student_state:
            raise RuntimeError("No student weights found in gathered checkpoint")
        # Write beside the target and swap, so no file of an earlier export survives in it.
        staging = Path(f"{save_path}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        extract_student_model(combined_model, tokenizer, str(staging), student_state)
        if training_record is not None:
            with open(staging / "training_record.json", "w") as f:
                json.dump(training_record, f, indent=2)
        shutil.rmtree(save_path, ignore_errors=True)
        staging.rename(save_path)
    trainer.runtime.barrier()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description="Train DMS adapter for Qwen3")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--prepare-dataset-only",
        action="store_true",
        help="Only prepare the dataset, then exit (run with single process first)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = TrainerSettings.from_config(cfg["trainer"])
    from .prepare_base import verify_prepared_base
    verify_prepared_base(cfg["model"]["name"])

    model_cfg = cfg["model"]
    dms_cfg = cfg["dms"]
    data_cfg = cfg["data"]

    # Build model arguments
    model_args = DistillationModelArguments(
        student=ModelArguments(
            model_name_or_path=model_cfg["name"],
            dtype=model_cfg.get("dtype", "float32"),
            forward_fn_kwargs=model_cfg.get("forward_fn_kwargs", {}),
        ),
        teacher=ModelArguments(
            model_name_or_path=model_cfg.get("teacher_name", model_cfg["name"]),
            dtype=model_cfg.get("teacher_dtype", model_cfg.get("dtype", "float32")),
            forward_fn_kwargs=model_cfg.get(
                "teacher_forward_fn_kwargs", model_cfg.get("forward_fn_kwargs", {})
            ),
        ),
    )

    # Resolve checkpoint resume
    checkpoint_path = resolve_checkpoint(cfg)

    logger.info(f"\n--- Config ---\n{yaml.dump(cfg, default_flow_style=False)}")

    # Tokenizer
    tokenizer = get_tokenizer(model_args.student)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = data_cfg.get("tokenizer_kwargs", {})
    configured_tokenizer = ConfiguredTokenizer(
        tokenizer=tokenizer,
        apply_chat_template_kwargs=tokenizer_kwargs,
        encode_kwargs={},
    )

    # Dataset
    train_dataset = _create_train_dataset(data_cfg, configured_tokenizer)

    if args.prepare_dataset_only:
        logger.info("Dataset preparation complete. Exiting (--prepare-dataset-only).")
        return

    from .check_env import check_environment
    check_environment(training_only=True)
    runtime = init_runtime()

    # Save config for reproducibility
    if runtime.is_main:
        save_config(cfg, settings.output_dir)

    # Build model
    trainer_state = DMSTrainerState()
    combined_model = build_combined_model(
        model_args=model_args,
        training_args=settings,
        dms_cfg=dms_cfg,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        trainer_state=trainer_state,
    )

    trainer = DMSTrainer(
        combined_model, train_dataset, settings, runtime, data_cfg["max_length"], trainer_state,
        resume_from=checkpoint_path,
    )

    # Train
    runtime_seconds = trainer.train()

    # Auto-save student model at end of training
    student_model_path = os.path.join(settings.output_dir, "student_model")
    export_trained_student(trainer, combined_model, tokenizer, student_model_path,
                           {**trainer.training_record(runtime_seconds), "config": cfg})

    logger.info("Training complete.")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
