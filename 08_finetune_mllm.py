#!/usr/bin/env python3
"""Executa fine-tuning parameter-efficient LoRA ou QLoRA de uma MLLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from mllm.data_utils import (
    REPRESENTATIONS,
    MLLMDataError,
    atomic_write_csv,
    atomic_write_json,
)
from mllm.model_utils import (
    DEFAULT_MODEL_ID,
    MLLMModelError,
    build_messages,
    hardware_information,
    load_mllm,
    package_versions,
    prepare_image,
    update_environment_file,
)
from mllm.prompts import CLASSIFICATION_PROMPT


class FineTuningError(ValueError):
    """Erro de configuracao ou integridade do fine-tuning."""


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


class MultimodalTrainingCollator:
    """Prepara imagem/texto e mascara o prompt para loss apenas da resposta."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, examples: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        images = [prepare_image(Path(str(example["image"]))) for example in examples]
        labels = [str(example["label"]) for example in examples]
        full_messages = [
            build_messages(image, assistant_label=label)
            for image, label in zip(images, labels, strict=True)
        ]
        prompt_messages = [build_messages(image) for image in images]

        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        if "input_ids" not in full_inputs or "attention_mask" not in full_inputs:
            raise FineTuningError("processor nao retornou input_ids/attention_mask")

        training_labels = full_inputs["input_ids"].clone()
        training_labels[full_inputs["attention_mask"] == 0] = -100
        for index in range(len(examples)):
            full_positions = torch.where(full_inputs["attention_mask"][index] != 0)[0]
            prompt_positions = torch.where(prompt_inputs["attention_mask"][index] != 0)[0]
            full_ids = full_inputs["input_ids"][index, full_positions]
            prompt_ids = prompt_inputs["input_ids"][index, prompt_positions]
            common_length = 0
            comparison_length = min(len(full_ids), len(prompt_ids))
            while (
                common_length < comparison_length
                and full_ids[common_length].item() == prompt_ids[common_length].item()
            ):
                common_length += 1
            if common_length == 0 or common_length >= len(full_ids):
                raise FineTuningError(
                    "nao foi possivel localizar os tokens da resposta no template"
                )
            training_labels[index, full_positions[:common_length]] = -100
            if not bool(torch.any(training_labels[index] != -100)):
                raise FineTuningError("exemplo sem tokens de resposta para calcular loss")
        full_inputs["labels"] = training_labels
        return dict(full_inputs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina somente adapters LoRA/QLoRA de uma MLLM, usando train "
            "para otimizacao e validation para selecao do melhor checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--train", type=Path, default=Path("data/mllm/train.jsonl"))
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/mllm/validation.jsonl"),
    )
    parser.add_argument(
        "--representation", choices=REPRESENTATIONS, default="pseudo_rgb"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/mllm_adapter")
    )
    parser.add_argument(
        "--environment-output",
        type=Path,
        default=Path("results/mllm_environment.json"),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Ativa QLoRA NF4; requer CUDA e bitsandbytes.",
    )
    parser.add_argument(
        "--target-modules",
        default=None,
        help="Modulos LoRA separados por virgula; por padrao detecta projecoes usuais.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--do-image-splitting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Usa crops adicionais do processor; deve coincidir com a inferencia.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    for argument_name in ("train", "validation"):
        path = getattr(args, argument_name)
        if not path.is_file():
            parser.error(f"--{argument_name} nao e arquivo acessivel: {path}")
    positive_values = {
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--batch-size": args.batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--lora-r": args.lora_r,
        "--lora-alpha": args.lora_alpha,
    }
    for name, value in positive_values.items():
        if not math.isfinite(float(value)) or value <= 0:
            parser.error(f"{name} deve ser finito e maior que zero")
    if not math.isfinite(args.lora_dropout) or not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout deve estar no intervalo [0, 1)")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience nao pode ser negativo")
    if args.use_4bit and not torch.cuda.is_available():
        parser.error(
            "--use-4bit requer CUDA; omita a opcao para usar LoRA sem quantizacao"
        )
    return args


def load_jsonl(path: Path, expected_representation: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FineTuningError(f"{path}:{line_number}: JSON invalido") from exc
            image_id = str(raw.get("image_id", "")).strip()
            image_path = Path(str(raw.get("image", ""))).expanduser().resolve()
            representation = str(raw.get("representation", ""))
            messages = raw.get("messages")
            if not image_id or not image_path.is_file():
                raise FineTuningError(
                    f"{path}:{line_number}: image_id/imagem invalido"
                )
            if representation != expected_representation:
                raise FineTuningError(
                    f"{path}:{line_number}: representacao {representation!r}, "
                    f"esperado {expected_representation!r}"
                )
            if not isinstance(messages, list) or len(messages) != 2:
                raise FineTuningError(f"{path}:{line_number}: messages invalido")
            if messages[0] != {"role": "user", "content": CLASSIFICATION_PROMPT}:
                raise FineTuningError(f"{path}:{line_number}: prompt divergente")
            assistant = messages[1]
            if not isinstance(assistant, dict):
                raise FineTuningError(f"{path}:{line_number}: resposta invalida")
            label = str(assistant.get("content", ""))
            if assistant.get("role") != "assistant" or label not in {
                "WATER",
                "NON_WATER",
            }:
                raise FineTuningError(f"{path}:{line_number}: label invalido")
            rows.append(
                {
                    "image_id": image_id,
                    "image": str(image_path),
                    "representation": representation,
                    "prompt": CLASSIFICATION_PROMPT,
                    "label": label,
                }
            )
    if not rows:
        raise FineTuningError(f"dataset vazio: {path}")
    ids = [str(row["image_id"]).casefold() for row in rows]
    if len(ids) != len(set(ids)):
        raise FineTuningError(f"image_id duplicado em {path}")
    if {str(row["label"]) for row in rows} != {"WATER", "NON_WATER"}:
        raise FineTuningError(f"{path} nao contem as duas classes")
    return rows


def detect_target_modules(model: torch.nn.Module) -> list[str]:
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    selected = [name for name in preferred if name in present]
    if not selected:
        selected = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
                and name.rsplit(".", 1)[-1] != "lm_head"
            }
        )
    if not selected:
        raise FineTuningError("nenhum modulo linear compativel com LoRA encontrado")
    return selected


def write_training_history(output_dir: Path, log_history: list[dict[str, object]]) -> None:
    rows = []
    for entry in log_history:
        if "loss" not in entry and "eval_loss" not in entry:
            continue
        rows.append(
            {
                "epoch": entry.get("epoch", ""),
                "step": entry.get("step", ""),
                "train_loss": entry.get("loss", ""),
                "validation_loss": entry.get("eval_loss", ""),
                "learning_rate": entry.get("learning_rate", ""),
            }
        )
    atomic_write_csv(
        output_dir / "training_history.csv",
        ["epoch", "step", "train_loss", "validation_loss", "learning_rate"],
        rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    environment_payload: dict[str, object] = {
        "status": "started",
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "representation": args.representation,
        "quantization": "QLoRA 4-bit NF4" if args.use_4bit else "LoRA (none)",
        "do_image_splitting": args.do_image_splitting,
    }
    update_environment_file(
        args.environment_output, "fine_tuning", environment_payload
    )
    try:
        train_rows = load_jsonl(args.train, args.representation)
        validation_rows = load_jsonl(args.validation, args.representation)
        train_ids = {str(row["image_id"]).casefold() for row in train_rows}
        validation_ids = {
            str(row["image_id"]).casefold() for row in validation_rows
        }
        if train_ids & validation_ids:
            raise FineTuningError("image_id aparece em train e validation")
        set_seed(args.seed)
        np.random.seed(args.seed)
        loaded = load_mllm(
            model_id=args.model_id,
            revision=args.revision,
            use_4bit=args.use_4bit,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            do_image_splitting=args.do_image_splitting,
        )
        base_model = loaded.model
        if args.use_4bit:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        target_modules = (
            [item.strip() for item in args.target_modules.split(",") if item.strip()]
            if args.target_modules
            else detect_target_modules(base_model)
        )
        if not target_modules:
            raise FineTuningError("--target-modules resultou em lista vazia")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, peft_config)
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        trainable, total = model.get_nb_trainable_parameters()
        if trainable <= 0 or trainable >= total:
            raise FineTuningError(
                f"parametros treinaveis invalidos para PEFT: {trainable}/{total}"
            )

        output_dir = args.output_dir.resolve()
        checkpoint_dir = output_dir / "checkpoints"
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            full_determinism=True,
            gradient_checkpointing=args.gradient_checkpointing,
            optim="adamw_torch",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            use_cpu=not torch.cuda.is_available(),
            dataloader_num_workers=0,
            dataloader_pin_memory=torch.cuda.is_available(),
            label_names=["labels"],
        )
        callbacks = []
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=JsonlDataset(train_rows),
            eval_dataset=JsonlDataset(validation_rows),
            data_collator=MultimodalTrainingCollator(loaded.processor),
            processing_class=loaded.processor,
            callbacks=callbacks,
        )
    except Exception as exc:
        environment_payload.update(
            {"status": "failed_before_training", "error": str(exc)}
        )
        update_environment_file(
            args.environment_output, "fine_tuning", environment_payload
        )
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        training_seconds = time.perf_counter() - started
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir, safe_serialization=True)
        write_training_history(output_dir, trainer.state.log_history)
        adapter_config_path = output_dir / "adapter_config.json"
        adapter_weights = output_dir / "adapter_model.safetensors"
        if not adapter_config_path.is_file() or not adapter_weights.is_file():
            raise FineTuningError("adapter PEFT nao foi salvo completamente")

        config = {
            "base_model": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision": loaded.resolved_revision,
            "representation": args.representation,
            "training_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples_used": 0,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": target_modules,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percentage": trainable / total * 100.0,
            "seed": args.seed,
            "quantization": "4-bit NF4" if args.use_4bit else "none",
            "gradient_checkpointing": args.gradient_checkpointing,
            "do_image_splitting": args.do_image_splitting,
            "early_stopping_patience": args.early_stopping_patience,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "training_seconds": training_seconds,
            "train_loss": float(result.training_loss),
            "prompt_sha256": hashlib.sha256(
                CLASSIFICATION_PROMPT.encode("utf-8")
            ).hexdigest(),
            "hardware": hardware_information(),
            "library_versions": package_versions(),
        }
        atomic_write_json(output_dir / "training_config.json", config)
    except Exception as exc:
        environment_payload.update(
            {
                "status": "failed_during_training",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        update_environment_file(
            args.environment_output, "fine_tuning", environment_payload
        )
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    environment_payload.update(
        {
            "status": "complete",
            "resolved_revision": loaded.resolved_revision,
            "training_samples": len(train_rows),
            "validation_samples": len(validation_rows),
            "test_samples_used": 0,
            "training_seconds": training_seconds,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "training_parameters": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "target_modules": target_modules,
                "seed": args.seed,
            },
        }
    )
    update_environment_file(
        args.environment_output, "fine_tuning", environment_payload
    )

    print("\nFine-tuning PEFT concluido")
    print(f"Modelo-base: {args.model_id}")
    print(f"Representacao: {args.representation}")
    print(f"Treino/validation/test usados: {len(train_rows)}/{len(validation_rows)}/0")
    print(
        f"Parametros treinaveis: {trainable}/{total} "
        f"({trainable / total * 100.0:.4f}%)"
    )
    print(f"Quantizacao: {'QLoRA 4-bit NF4' if args.use_4bit else 'LoRA'}")
    print(f"Melhor checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Melhor validation loss: {trainer.state.best_metric}")
    print(f"Tempo total: {training_seconds:.2f}s")
    print(f"Adapter: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
