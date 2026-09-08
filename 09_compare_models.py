#!/usr/bin/env python3
"""Compara Random Forest e MLLM no mesmo conjunto geografico de teste."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/artigo_llm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation.metrics import (
    EvaluationError,
    Prediction,
    calculate_metrics,
    confusion_counts,
    load_ground_truth,
    load_predictions,
    validate_prediction_sets,
)
from mllm.prompts import CLASSIFICATION_PROMPT


MODEL_COMPARISON_FIELDS = [
    "model",
    "training",
    "representation",
    "evaluation_view",
    "samples",
    "evaluated_samples",
    "valid_predictions",
    "invalid_predictions",
    "invalid_prediction_rate",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "f1_ci_low",
    "f1_ci_high",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "metric_notes",
]
ALL_PREDICTION_FIELDS = [
    "image_id",
    "event",
    "true_class",
    "true_class_id",
    "model",
    "training",
    "representation",
    "predicted_class",
    "prediction_valid",
    "raw_response",
    "probability_non_water",
    "probability_water",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara Random Forest, MLLM zero-shot e MLLM fine-tuned com "
            "metricas uniformes no mesmo test.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument(
        "--random-forest-results",
        type=Path,
        default=Path("results/random_forest"),
    )
    parser.add_argument(
        "--random-forest-metadata",
        type=Path,
        default=Path("models/random_forest_features.json"),
    )
    parser.add_argument(
        "--zero-shot-results",
        type=Path,
        default=Path("results/mllm_zero_shot"),
    )
    parser.add_argument(
        "--finetuned-results",
        type=Path,
        default=Path("results/mllm_finetuned"),
    )
    parser.add_argument(
        "--finetuning-config",
        type=Path,
        default=Path("models/mllm_adapter_pseudo_rgb/training_config.json"),
    )
    parser.add_argument(
        "--mllm-environment",
        type=Path,
        default=Path("results/mllm_environment.json"),
    )
    parser.add_argument("--representation", default="pseudo_rgb")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/final")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    required_files = {
        "--train": args.train,
        "--test": args.test,
        "Random Forest": args.random_forest_results / "test_predictions.csv",
        "MLLM zero-shot": args.zero_shot_results / "test_predictions.csv",
        "MLLM fine-tuned": args.finetuned_results / "test_predictions.csv",
        "--random-forest-metadata": args.random_forest_metadata,
        "--finetuning-config": args.finetuning_config,
        "--mllm-environment": args.mllm_environment,
    }
    for label, path in required_files.items():
        if not path.is_file():
            parser.error(f"{label}: arquivo inexistente: {path}")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples nao pode ser negativo")
    return args


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                normalized = {
                    key: "NaN"
                    if isinstance(value, float) and math.isnan(value)
                    else value
                    for key, value in row.items()
                }
                writer.writerow(normalized)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"JSON invalido {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"objeto JSON esperado em {path}")
    return value


def _validate_no_leakage(args: argparse.Namespace) -> dict[str, object]:
    train = load_ground_truth(args.train, expected_split="train")
    test = load_ground_truth(args.test, expected_split="test")
    train_ids = {row.image_id for row in train}
    test_ids = {row.image_id for row in test}
    train_events = {row.event for row in train}
    test_events = {row.event for row in test}
    if train_ids & test_ids:
        raise EvaluationError("image_id compartilhado entre treino e teste")
    if train_events & test_events:
        raise EvaluationError("evento compartilhado entre treino e teste")

    rf_metadata = _load_json(args.random_forest_metadata)
    feature_names = [str(name).casefold() for name in rf_metadata.get("feature_names", [])]
    forbidden = ("water_percentage", "labelhand", "mask", "class", "target")
    if not feature_names or any(
        token in feature for feature in feature_names for token in forbidden
    ):
        raise EvaluationError("features do Random Forest ausentes ou com leakage")
    feature_source = str(rf_metadata.get("feature_source", "")).casefold()
    if "sentinel-1" not in feature_source:
        raise EvaluationError("fonte das features do Random Forest nao comprovada")

    finetuning = _load_json(args.finetuning_config)
    if int(finetuning.get("test_samples_used", -1)) != 0:
        raise EvaluationError("configuracao LoRA indica uso do conjunto de teste")
    prompt_hash = hashlib.sha256(CLASSIFICATION_PROMPT.encode("utf-8")).hexdigest()
    if finetuning.get("prompt_sha256") != prompt_hash:
        raise EvaluationError("hash do prompt de fine-tuning diverge do prompt atual")
    lowered_prompt = CLASSIFICATION_PROMPT.casefold()
    if "water_percentage" in lowered_prompt or "labelhand" in lowered_prompt:
        raise EvaluationError("prompt MLLM contem informacao de ground truth")

    environment = _load_json(args.mllm_environment)
    runs = environment.get("runs", {})
    if not isinstance(runs, dict):
        raise EvaluationError("registro de execucoes MLLM invalido")
    for run_name in ("zero_shot", "finetuned"):
        run = runs.get(run_name)
        if not isinstance(run, dict) or run.get("status") != "complete":
            raise EvaluationError(f"execucao MLLM incompleta: {run_name}")
        if run.get("prompt_sha256") != prompt_hash:
            raise EvaluationError(f"prompt divergente na execucao {run_name}")
        if run.get("representation") != args.representation:
            raise EvaluationError(f"representacao divergente na execucao {run_name}")
    return {
        "train_test_id_overlap": 0,
        "train_test_event_overlap": 0,
        "test_samples_used_by_lora": 0,
        "prompt_sha256": prompt_hash,
        "forbidden_inputs": [],
    }


def _save_confusion(
    predictions: Sequence[Prediction], title: str, path: Path
) -> None:
    matrix = confusion_counts(predictions)
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["NON_WATER", "WATER"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    invalid = sum(not row.prediction_valid for row in predictions)
    axis.set_title(f"{title}\nInvalid predictions: {invalid}")
    figure.tight_layout()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def compare(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[Prediction]]:
    validation = _validate_no_leakage(args)
    truth = load_ground_truth(args.test)
    specifications = [
        (
            "Random Forest",
            "supervised",
            "vv_vh_features",
            args.random_forest_results / "test_predictions.csv",
        ),
        (
            "MLLM Zero-Shot",
            "zero-shot",
            args.representation,
            args.zero_shot_results / "test_predictions.csv",
        ),
        (
            "MLLM Fine-Tuned",
            "fine-tuned",
            args.representation,
            args.finetuned_results / "test_predictions.csv",
        ),
    ]
    prediction_sets: list[list[Prediction]] = []
    comparison_rows: list[dict[str, object]] = []
    for model, training, representation, path in specifications:
        predictions = load_predictions(
            path,
            truth,
            model=model,
            training=training,
            representation=representation,
        )
        prediction_sets.append(predictions)
        for view in ("valid-only", "strict"):
            metric = calculate_metrics(
                predictions,
                view,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            comparison_rows.append(
                {
                    "model": model,
                    "training": training,
                    "representation": representation,
                    **metric.to_dict(),
                }
            )
    validate_prediction_sets(prediction_sets)

    output_dir = args.output_dir.resolve()
    _atomic_csv(
        output_dir / "model_comparison.csv",
        MODEL_COMPARISON_FIELDS,
        comparison_rows,
    )
    flattened = [row for rows in prediction_sets for row in rows]
    consolidated_rows = [
        {
            **row.__dict__,
            "predicted_class": row.predicted_class or "",
            "probability_non_water": (
                row.probability_non_water
                if row.probability_non_water is not None
                else ""
            ),
            "probability_water": (
                row.probability_water if row.probability_water is not None else ""
            ),
        }
        for row in flattened
    ]
    _atomic_csv(
        output_dir / "all_test_predictions.csv",
        ALL_PREDICTION_FIELDS,
        consolidated_rows,
    )
    confusion_dir = output_dir / "confusion_matrices"
    filenames = (
        "random_forest.png",
        f"mllm_zero_shot_{args.representation}.png",
        f"mllm_fine_tuned_{args.representation}.png",
    )
    for predictions, specification, filename in zip(
        prediction_sets, specifications, filenames, strict=True
    ):
        _save_confusion(predictions, specification[0], confusion_dir / filename)
    validation["test_ids_equal_across_models"] = True
    validation["true_labels_equal_across_models"] = True
    validation["test_samples"] = len(truth)
    _atomic_json(output_dir / "comparison_validation.json", validation)
    return comparison_rows, flattened


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, _ = compare(args)
    except Exception as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1
    print("\nComparacao principal concluida")
    for row in rows:
        if row["evaluation_view"] != "strict":
            continue
        print(
            f"{row['model']}: accuracy={row['accuracy']:.6f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.6f}, "
            f"f1={row['f1']:.6f}, invalidas={row['invalid_predictions']}"
        )
    print(f"Saida: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
