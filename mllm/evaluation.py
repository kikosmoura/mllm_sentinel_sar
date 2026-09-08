"""Rotina unica de avaliacao para a MLLM original e com adapter PEFT."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from .data_utils import (
    CLASS_TO_ID,
    REPRESENTATIONS,
    MLLMDataError,
    MLLMExample,
    atomic_write_csv,
    load_split,
    validate_geographic_isolation,
)
from .model_utils import (
    DEFAULT_MODEL_ID,
    MLLMModelError,
    generate_response,
    load_mllm,
    parse_predicted_class,
    update_environment_file,
)
from .prompts import CLASSIFICATION_PROMPT


PREDICTION_FIELDS = [
    "image_id",
    "event",
    "representation",
    "true_class",
    "true_class_id",
    "raw_response",
    "predicted_class",
    "prediction_valid",
]
METRIC_FIELDS = [
    "model",
    "adapter",
    "split",
    "representation",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "invalid_responses",
    "samples",
    "inference_seconds",
]


def create_evaluation_parser(
    description: str,
    default_output_dir: str,
    require_adapter: bool,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Checkpoint-base multimodal no Hugging Face Hub ou caminho local.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revisao/commit opcional do modelo-base.",
    )
    parser.add_argument(
        "--train-audit",
        type=Path,
        default=Path("data/splits/train.csv"),
        help=(
            "Split de treino lido apenas para auditar leakage geografico; "
            "nunca e enviado ao modelo nem usado como contexto."
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
        help="Split usado para inferencia de validacao.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/splits/test.csv"),
        help="Split usado para inferencia final de teste.",
    )
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="pseudo_rgb",
        help="Representacao SAR de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(default_output_dir),
        help="Diretorio das predicoes e metricas.",
    )
    parser.add_argument(
        "--environment-output",
        type=Path,
        default=Path("results/mllm_environment.json"),
        help="Registro compartilhado de ambiente e execucoes MLLM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Limite curto para a resposta classificatoria.",
    )
    parser.add_argument(
        "--do-image-splitting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Permite crops adicionais do processor. Desativado por padrao "
            "porque os chips SAR ja sao quadrados 512x512."
        ),
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Carrega o modelo-base em 4 bits (requer CUDA e bitsandbytes).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Permite codigo remoto do checkpoint; desativado por padrao.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Impede downloads e usa somente arquivos no cache local.",
    )
    if require_adapter:
        parser.add_argument(
            "--adapter-path",
            type=Path,
            required=True,
            help="Diretorio do adapter PEFT/LoRA treinado.",
        )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens deve ser maior que zero")
    for argument_name in ("train_audit", "validation", "test"):
        path = getattr(args, argument_name)
        if not path.is_file():
            parser.error(
                f"--{argument_name.replace('_', '-')} nao e arquivo acessivel: {path}"
            )
    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path is not None and not adapter_path.is_dir():
        parser.error(f"--adapter-path nao e diretorio acessivel: {adapter_path}")


def control_metrics(
    examples: Sequence[MLLMExample],
    predictions: Sequence[dict[str, object]],
) -> dict[str, float | int]:
    if len(examples) != len(predictions):
        raise MLLMDataError(
            f"predicoes={len(predictions)} difere de amostras={len(examples)}"
        )
    true_water = sum(example.class_id == 1 for example in examples)
    true_non_water = len(examples) - true_water
    tp = tn = fp = 0
    invalid = 0
    for example, prediction in zip(examples, predictions, strict=True):
        predicted_class = str(prediction["predicted_class"])
        if not bool(prediction["prediction_valid"]):
            invalid += 1
            continue
        predicted_id = CLASS_TO_ID[predicted_class]
        if example.class_id == 1 and predicted_id == 1:
            tp += 1
        elif example.class_id == 0 and predicted_id == 0:
            tn += 1
        elif example.class_id == 0 and predicted_id == 1:
            fp += 1

    accuracy = (tp + tn) / len(examples)
    recall = tp / true_water if true_water else math.nan
    specificity = tn / true_non_water if true_non_water else math.nan
    balanced_accuracy = (recall + specificity) / 2.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "invalid_responses": invalid,
        "samples": len(examples),
    }


def evaluate_examples(
    loaded: object,
    examples: Sequence[MLLMExample],
    max_new_tokens: int,
) -> tuple[list[dict[str, object]], float, list[str]]:
    predictions: list[dict[str, object]] = []
    errors: list[str] = []
    started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        try:
            raw_response = generate_response(
                loaded, example.image_path, max_new_tokens=max_new_tokens
            )
        except Exception as exc:
            raw_response = f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
            errors.append(f"{example.split}/{example.image_id}: {exc}")
        predicted_class = parse_predicted_class(raw_response)
        predictions.append(
            {
                "image_id": example.image_id,
                "event": example.event,
                "representation": example.representation,
                "true_class": example.class_name,
                "true_class_id": example.class_id,
                "raw_response": raw_response,
                "predicted_class": predicted_class or "",
                "prediction_valid": predicted_class is not None,
            }
        )
        if index % 10 == 0 or index == len(examples):
            print(
                f"{example.split}: {index}/{len(examples)} inferencias concluidas",
                flush=True,
            )
    return predictions, time.perf_counter() - started, errors


def run_evaluation(args: argparse.Namespace, run_name: str) -> int:
    adapter_path = getattr(args, "adapter_path", None)
    environment_base = {
        "status": "started",
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "adapter_path": str(adapter_path.resolve()) if adapter_path else None,
        "representation": args.representation,
        "prompt_sha256": hashlib.sha256(
            CLASSIFICATION_PROMPT.encode("utf-8")
        ).hexdigest(),
        "deterministic_generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "do_image_splitting": args.do_image_splitting,
    }
    update_environment_file(args.environment_output, run_name, environment_base)

    try:
        splits = {
            "train": load_split(
                args.train_audit, "train", args.representation, validate_images=False
            ),
            "validation": load_split(
                args.validation, "validation", args.representation
            ),
            "test": load_split(args.test, "test", args.representation),
        }
        validate_geographic_isolation(splits)
        loaded = load_mllm(
            model_id=args.model_id,
            revision=args.revision,
            adapter_path=adapter_path,
            use_4bit=args.use_4bit,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            do_image_splitting=args.do_image_splitting,
        )
    except Exception as exc:
        environment_base.update(
            {"status": "failed_before_inference", "error": str(exc)}
        )
        update_environment_file(args.environment_output, run_name, environment_base)
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    all_metrics: list[dict[str, object]] = []
    errors: list[str] = []
    output_dir = args.output_dir.resolve()
    total_started = time.perf_counter()
    try:
        for split_name in ("validation", "test"):
            predictions, elapsed, split_errors = evaluate_examples(
                loaded, splits[split_name], args.max_new_tokens
            )
            errors.extend(split_errors)
            atomic_write_csv(
                output_dir / f"{split_name}_predictions.csv",
                PREDICTION_FIELDS,
                predictions,
            )
            metrics = control_metrics(splits[split_name], predictions)
            all_metrics.append(
                {
                    "model": args.model_id,
                    "adapter": str(adapter_path.resolve()) if adapter_path else "",
                    "split": split_name,
                    "representation": args.representation,
                    **metrics,
                    "inference_seconds": elapsed,
                }
            )
        atomic_write_csv(output_dir / "metrics.csv", METRIC_FIELDS, all_metrics)
    except Exception as exc:
        environment_base.update({"status": "failed_during_inference", "error": str(exc)})
        update_environment_file(args.environment_output, run_name, environment_base)
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    total_elapsed = time.perf_counter() - total_started
    environment_base.update(
        {
            "status": "complete",
            "resolved_revision": loaded.resolved_revision,
            "device": loaded.device,
            "dtype": loaded.dtype,
            "do_image_splitting": loaded.do_image_splitting,
            "samples": {
                name: len(examples) for name, examples in splits.items()
            },
            "total_inference_seconds": total_elapsed,
            "invalid_responses": {
                str(row["split"]): int(row["invalid_responses"])
                for row in all_metrics
            },
            "inference_errors": errors,
        }
    )
    update_environment_file(args.environment_output, run_name, environment_base)

    print("\nMetricas de controle")
    for row in all_metrics:
        print(
            f"{row['split']}: accuracy={row['accuracy']:.6f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}, "
            f"precision={row['precision']:.6f}, recall={row['recall']:.6f}, "
            f"f1={row['f1']:.6f}, invalidas={row['invalid_responses']}"
        )
    if errors:
        print("\nRespostas com erro de inferencia:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    print(f"Resultados: {output_dir}")
    return 0
