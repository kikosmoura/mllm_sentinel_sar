#!/usr/bin/env python3
"""Gera tabelas, graficos e manifesto final a partir de predicoes validadas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/artigo_llm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from evaluation.comparison import (
    CNN_MODEL, DISPLAY_NAMES, MAIN_REPRESENTATION, display_name,
    main_model_names, training_regimes, validate_cnn_artifacts,
)


REPRESENTATIONS = ("vv", "vh", "pseudo_rgb")
ABLATION_FIELDS = [
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
GAIN_FIELDS = [
    "representation",
    "evaluation_view",
    "zero_shot_accuracy",
    "finetuned_accuracy",
    "delta_accuracy",
    "zero_shot_balanced_accuracy",
    "finetuned_balanced_accuracy",
    "delta_balanced_accuracy",
    "zero_shot_f1",
    "finetuned_f1",
    "delta_f1",
]
MAIN_TABLE_FIELDS = [
    "Model",
    "SAR Representation",
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1",
    "Invalid Predictions",
    "Invalid Prediction Rate",
    "ROC-AUC",
    "F1 CI Low",
    "F1 CI High",
    "Balanced Accuracy CI Low",
    "Balanced Accuracy CI High",
]
ABLATION_TABLE_FIELDS = [
    "Representation",
    "Zero-Shot F1",
    "Fine-Tuned F1",
    "F1 Gain",
    "Zero-Shot Balanced Accuracy",
    "Fine-Tuned Balanced Accuracy",
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
FIXED_CONFIG_KEYS = (
    "base_model",
    "requested_revision",
    "training_samples",
    "validation_samples",
    "test_samples_used",
    "epochs",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "seed",
    "quantization",
    "gradient_checkpointing",
    "do_image_splitting",
    "early_stopping_patience",
    "prompt_sha256",
)


class FinalResultsError(RuntimeError):
    """Inconsistencia nos artefatos finais do experimento."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolida comparacao e ablation study em tabelas, figuras, "
            "matrizes de confusao e manifesto reprodutivel."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/splits/validation.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--include-cnn", action="store_true")
    parser.add_argument("--cnn-results", type=Path)
    parser.add_argument("--cnn-config", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Padrao final_with_resnet18 com CNN; final no modo antigo"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.results_dir / (
        "final_with_resnet18" if args.include_cnn else "final")
    args.cnn_results = args.cnn_results or args.results_dir / "resnet18_pseudo_rgb"
    args.cnn_config = args.cnn_config or args.models_dir / "resnet18_pseudo_rgb/training_config.json"
    if args.include_cnn and args.output_dir.resolve() == (args.results_dir / "final").resolve():
        parser.error("use outra saida para preservar results/final")
    for name in ("train", "validation", "test"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name}: arquivo inexistente: {getattr(args, name)}")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples nao pode ser negativo")
    return args


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalResultsError(f"JSON invalido {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalResultsError(f"objeto JSON esperado em {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FinalResultsError(f"CSV inexistente: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: "NaN"
                        if isinstance(value, float) and math.isnan(value)
                        else value
                        for key, value in row.items()
                    }
                )
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


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "numpy",
        "scikit-learn",
        "matplotlib",
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "Pillow",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _load_main_strict(path: Path, include_cnn: bool = False) -> list[dict[str, str]]:
    rows = _read_csv(path)
    ordered_models = main_model_names(include_cnn)
    required_models = set(ordered_models)
    strict = [row for row in rows if row.get("evaluation_view") == "strict"]
    if {row.get("model") for row in strict} != required_models:
        raise FinalResultsError(f"model_comparison.csv deve conter {ordered_models}")
    if len(strict) != len(required_models):
        raise FinalResultsError("model_comparison.csv possui linhas strict duplicadas")
    expected = {(model, view) for model in ordered_models for view in ("strict", "valid-only")}
    if len(rows) != len(expected) or {(r["model"], r["evaluation_view"]) for r in rows} != expected:
        raise FinalResultsError("model_comparison.csv possui visoes ausentes/duplicadas")
    for row in strict:
        representation = "vv_vh_features" if row["model"] == "Random Forest" else MAIN_REPRESENTATION
        if row["representation"] != representation:
            raise FinalResultsError("representacao principal deve ser fixada em pseudo_rgb")
    indexed = {row["model"]: row for row in strict}
    return [indexed[model] for model in ordered_models]


def _validate_main_metrics(path: Path, prediction_sets: dict[str, list[Prediction]],
                           ablation_metrics: dict, bootstrap_samples: int, seed: int) -> None:
    """Evita consolidar uma tabela desatualizada ou calculada com outra configuracao."""
    indexed = {(row["model"], row["evaluation_view"]): row for row in _read_csv(path)}
    for model, predictions in prediction_sets.items():
        for view in ("valid-only", "strict"):
            if model in ("MLLM Zero-Shot", "MLLM Fine-Tuned"):
                training = "zero-shot" if model == "MLLM Zero-Shot" else "fine-tuned"
                expected = ablation_metrics[(training, MAIN_REPRESENTATION, view)]
            else:
                expected = calculate_metrics(predictions, view,
                                             bootstrap_samples=bootstrap_samples, seed=seed).to_dict()
            actual = indexed[(model, view)]
            for key, value in expected.items():
                if key in ("model", "training", "representation"):
                    continue
                if isinstance(value, (float, int)):
                    candidate = float(actual[key])
                    matches = (math.isnan(float(value)) and math.isnan(candidate)) or math.isclose(
                        candidate, float(value), rel_tol=0.0, abs_tol=1e-12)
                else:
                    matches = actual[key] == value
                if not matches:
                    raise FinalResultsError(f"metrica principal divergente: {model}/{view}/{key}")


def _validate_splits(args: argparse.Namespace) -> dict[str, list[object]]:
    splits = {
        "train": load_ground_truth(args.train, expected_split="train"),
        "validation": load_ground_truth(
            args.validation, expected_split="validation"
        ),
        "test": load_ground_truth(args.test, expected_split="test"),
    }
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if {row.image_id for row in splits[left]} & {
                row.image_id for row in splits[right]
            }:
                raise FinalResultsError(f"IDs compartilhados em {left}/{right}")
            if {row.event for row in splits[left]} & {
                row.event for row in splits[right]
            }:
                raise FinalResultsError(f"eventos compartilhados em {left}/{right}")
    return splits


def _adapter_directories(models_dir: Path) -> dict[str, Path]:
    return {
        "vv": models_dir / "mllm_adapter_vv",
        "vh": models_dir / "mllm_adapter_vh",
        "pseudo_rgb": models_dir / "mllm_adapter_pseudo_rgb",
    }


def _validate_training_inputs(
    args: argparse.Namespace,
    splits: dict[str, list[object]],
) -> dict[str, dict[str, object]]:
    prompt_hash = hashlib.sha256(CLASSIFICATION_PROMPT.encode("utf-8")).hexdigest()
    lowered_prompt = CLASSIFICATION_PROMPT.casefold()
    if any(token in lowered_prompt for token in ("water_percentage", "labelhand")):
        raise FinalResultsError("prompt contem dado proibido")
    test_ids = {row.image_id for row in splits["test"]}
    configs: dict[str, dict[str, object]] = {}
    jsonl_directories = {
        "vv": Path("data/mllm_ablation/vv"),
        "vh": Path("data/mllm_ablation/vh"),
        "pseudo_rgb": Path("data/mllm"),
    }
    for representation in REPRESENTATIONS:
        adapter_dir = _adapter_directories(args.models_dir.resolve())[representation]
        config = _read_json(adapter_dir / "training_config.json")
        if config.get("representation") != representation:
            raise FinalResultsError(
                f"representacao incorreta no adapter {representation}"
            )
        if int(config.get("test_samples_used", -1)) != 0:
            raise FinalResultsError(f"teste usado no adapter {representation}")
        if config.get("prompt_sha256") != prompt_hash:
            raise FinalResultsError(f"prompt divergente no adapter {representation}")
        configs[representation] = config

        for split_name in ("train", "validation"):
            jsonl_path = jsonl_directories[representation] / f"{split_name}.jsonl"
            if not jsonl_path.is_file():
                raise FinalResultsError(f"JSONL inexistente: {jsonl_path}")
            records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            if len(records) != len(splits[split_name]):
                raise FinalResultsError(f"contagem incorreta em {jsonl_path}")
            for record in records:
                if record.get("image_id") in test_ids:
                    raise FinalResultsError(f"ID de teste presente em {jsonl_path}")
                if record.get("representation") != representation:
                    raise FinalResultsError(f"representacao divergente em {jsonl_path}")
                image_path = Path(str(record.get("image", ""))).resolve()
                if not image_path.is_file() or image_path.parent.name != representation:
                    raise FinalResultsError(f"imagem/representacao invalida em {jsonl_path}")
                messages = record.get("messages")
                if (
                    not isinstance(messages, list)
                    or len(messages) != 2
                    or messages[0]
                    != {"role": "user", "content": CLASSIFICATION_PROMPT}
                ):
                    raise FinalResultsError(f"prompt invalido em {jsonl_path}")

    reference = configs["pseudo_rgb"]
    for representation in ("vv", "vh"):
        differences = [
            key
            for key in FIXED_CONFIG_KEYS
            if configs[representation].get(key) != reference.get(key)
        ]
        if differences:
            raise FinalResultsError(
                f"fatores alem da representacao mudaram em {representation}: {differences}"
            )
    return configs


def _load_ablation_predictions(
    args: argparse.Namespace, truth: Sequence[object]
) -> dict[tuple[str, str], list[Prediction]]:
    ablation_dir = args.results_dir.resolve() / "ablation"
    result: dict[tuple[str, str], list[Prediction]] = {}
    for representation in REPRESENTATIONS:
        for training, prefix in (
            ("zero-shot", "zero_shot"),
            ("fine-tuned", "finetuned"),
        ):
            result[(training, representation)] = load_predictions(
                ablation_dir / f"{prefix}_{representation}.csv",
                truth,
                model="MLLM",
                training=training,
                representation=representation,
            )
    validate_prediction_sets(result.values())
    return result


def _metric_rows(
    prediction_sets: dict[tuple[str, str], list[Prediction]],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    index: dict[tuple[str, str, str], dict[str, object]] = {}
    for (training, representation), predictions in prediction_sets.items():
        for view in ("valid-only", "strict"):
            metric = calculate_metrics(
                predictions,
                view,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
            row = {
                "model": "MLLM",
                "training": training,
                "representation": representation,
                **metric.to_dict(),
            }
            rows.append(row)
            index[(training, representation, view)] = row
    return rows, index


def _gain_rows(
    metrics: dict[tuple[str, str, str], dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for representation in REPRESENTATIONS:
        for view in ("valid-only", "strict"):
            zero = metrics[("zero-shot", representation, view)]
            fine = metrics[("fine-tuned", representation, view)]
            rows.append(
                {
                    "representation": representation,
                    "evaluation_view": view,
                    "zero_shot_accuracy": zero["accuracy"],
                    "finetuned_accuracy": fine["accuracy"],
                    "delta_accuracy": float(fine["accuracy"])
                    - float(zero["accuracy"]),
                    "zero_shot_balanced_accuracy": zero["balanced_accuracy"],
                    "finetuned_balanced_accuracy": fine["balanced_accuracy"],
                    "delta_balanced_accuracy": float(fine["balanced_accuracy"])
                    - float(zero["balanced_accuracy"]),
                    "zero_shot_f1": zero["f1"],
                    "finetuned_f1": fine["f1"],
                    "delta_f1": float(fine["f1"]) - float(zero["f1"]),
                }
            )
    return rows


def _save_confusion_matrices(
    output_dir: Path,
    base_predictions: dict[str, list[Prediction]],
    ablation: dict[tuple[str, str], list[Prediction]],
) -> None:
    filenames = {"Random Forest": "random_forest.png", CNN_MODEL: "resnet18_pseudo_rgb.png"}
    specifications = [(display_name(model), filenames[model], predictions)
                      for model, predictions in base_predictions.items()]
    for training, representation in ablation:
        specifications.append(
            (
                f"{'MLLM + LoRA' if training == 'fine-tuned' else 'MLLM zero-shot'} - {representation}",
                f"mllm_{training.replace('-', '_')}_{representation}.png",
                ablation[(training, representation)],
            )
        )
    for title, filename, predictions in specifications:
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
        _save_figure(figure, output_dir / "confusion_matrices" / filename)


def _main_metrics_figure(main_rows: Sequence[dict[str, str]], path: Path) -> None:
    metrics = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")
    labels = ("Accuracy", "Balanced accuracy", "Precision", "Recall", "F1")
    models = [display_name(row["model"]) for row in main_rows]
    x = np.arange(len(metrics))
    width = 0.8 / len(main_rows)
    figure, axis = plt.subplots(figsize=(max(9.0, 2.4 * len(main_rows)), 5.2))
    for index, row in enumerate(main_rows):
        axis.bar(
            x + (index - (len(main_rows) - 1) / 2) * width,
            [float(row[name]) for name in metrics],
            width,
            label=models[index],
            color=plt.get_cmap("tab10")(index),
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Test metrics by model (strict)")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=min(2, len(main_rows)), frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _model_f1_figure(main_rows: Sequence[dict[str, str]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(7.0, 2.4 * len(main_rows)), 4.8))
    names = [display_name(row["model"]) for row in main_rows]
    values = [float(row["f1"]) for row in main_rows]
    bars = axis.bar(names, values, color=[plt.get_cmap("tab10")(i) for i in range(len(names))])
    for index, row in enumerate(main_rows):
        low, high = float(row["f1_ci_low"]), float(row["f1_ci_high"])
        if math.isfinite(low) and math.isfinite(high):
            axis.vlines(index, low, high, color="black", linewidth=1.4)
            axis.plot([index, index], [low, high], linestyle="none", marker="_", color="black")
        axis.text(index, values[index] + 0.015, f"{values[index]:.3f}", ha="center",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("F1")
    axis.set_title("Test F1 comparison (strict)\n95% bootstrap intervals by image")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _representation_figure(
    metrics: dict[tuple[str, str, str], dict[str, object]], path: Path
) -> None:
    x = np.arange(len(REPRESENTATIONS))
    width = 0.36
    zero = [float(metrics[("zero-shot", rep, "strict")]["f1"]) for rep in REPRESENTATIONS]
    fine = [float(metrics[("fine-tuned", rep, "strict")]["f1"]) for rep in REPRESENTATIONS]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    bars_zero = axis.bar(x - width / 2, zero, width, label="Zero-shot")
    bars_fine = axis.bar(x + width / 2, fine, width, label="MLLM + LoRA")
    axis.bar_label(bars_zero, fmt="%.3f", padding=3)
    axis.bar_label(bars_fine, fmt="%.3f", padding=3)
    axis.set_xticks(x, [rep.upper() if rep != "pseudo_rgb" else "pseudo-RGB" for rep in REPRESENTATIONS])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("F1")
    axis.set_title("SAR representation ablation (strict)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _gain_figure(gain_rows: Sequence[dict[str, object]], path: Path) -> None:
    strict = [row for row in gain_rows if row["evaluation_view"] == "strict"]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    labels = [
        str(row["representation"]).upper()
        if row["representation"] != "pseudo_rgb"
        else "pseudo-RGB"
        for row in strict
    ]
    values = [float(row["delta_f1"]) for row in strict]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in values]
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, fmt="%+.3f", padding=3)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("MLLM + LoRA F1 - zero-shot F1")
    axis.set_title("LoRA gain by SAR representation (strict)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _consolidated_rows(
    base_predictions: dict[str, list[Prediction]],
    ablation: dict[tuple[str, str], list[Prediction]],
) -> list[dict[str, object]]:
    all_predictions = [row for predictions in base_predictions.values() for row in predictions]
    for predictions in ablation.values():
        all_predictions.extend(predictions)
    seen = set()
    for row in all_predictions:
        family = "MLLM" if row.model.startswith("MLLM") else row.model
        key = (family, row.training, row.representation, row.image_id)
        if key in seen:
            raise FinalResultsError(f"condicao/amostra duplicada na consolidacao: {key}")
        seen.add(key)
    return [
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
        for row in all_predictions
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        splits = _validate_splits(args)
        configs = _validate_training_inputs(args, splits)
        output_dir = args.output_dir.resolve()
        main_rows = _load_main_strict(output_dir / "model_comparison.csv", args.include_cnn)
        comparison_validation = _read_json(output_dir / "comparison_validation.json")
        if (comparison_validation.get("seed", args.seed) != args.seed
                or comparison_validation.get("bootstrap_samples", args.bootstrap_samples)
                != args.bootstrap_samples):
            raise FinalResultsError("comparacao e consolidacao usam seed/bootstrap diferentes")
        ablation = _load_ablation_predictions(args, splits["test"])
        metric_rows, metric_index = _metric_rows(
            ablation, args.bootstrap_samples, args.seed
        )
        gain_rows = _gain_rows(metric_index)

        rf_predictions = load_predictions(
            args.results_dir.resolve() / "random_forest/test_predictions.csv",
            splits["test"],
            model="Random Forest",
            training="supervised",
            representation="vv_vh_features",
        )
        base_predictions = {"Random Forest": rf_predictions}
        cnn_config = None
        if args.include_cnn:
            cnn_config, cnn_validation = validate_cnn_artifacts(
                args.cnn_config, args.cnn_results,
                {name: getattr(args, name) for name in splits})
            if (cnn_config["training_samples"] != len(splits["train"])
                    or cnn_config["validation_samples"] != len(splits["validation"])):
                raise FinalResultsError("CNN: contagens dos splits divergentes")
            comparison_validation.update(cnn_validation)
            base_predictions[CNN_MODEL] = load_predictions(
                args.cnn_results / "test_predictions.csv", splits["test"],
                model=CNN_MODEL, training="supervised-transfer-learning",
                representation=MAIN_REPRESENTATION)
        main_predictions = dict(base_predictions)
        for model, training, directory in (
            ("MLLM Zero-Shot", "zero-shot", "mllm_zero_shot"),
            ("MLLM Fine-Tuned", "fine-tuned", "mllm_finetuned"),
        ):
            predictions = load_predictions(
                args.results_dir / directory / "test_predictions.csv", splits["test"],
                model=model, training=training, representation=MAIN_REPRESENTATION)
            # A condicao pseudo_rgb da ablation e a mesma da comparacao principal.
            # O nome interno "MLLM" da ablation e preservado no CSV consolidado.
            for main_prediction, ablation_prediction in zip(
                    predictions, ablation[(training, MAIN_REPRESENTATION)], strict=True):
                candidate = {**main_prediction.__dict__, "model": "MLLM"}
                if candidate != ablation_prediction.__dict__:
                    raise FinalResultsError(f"predicoes principais/ablation divergentes: {model}")
            main_predictions[model] = predictions
        validate_prediction_sets([*main_predictions.values(), *ablation.values()])
        _validate_main_metrics(output_dir / "model_comparison.csv", main_predictions,
                               metric_index, args.bootstrap_samples, args.seed)

        _atomic_csv(output_dir / "ablation_results.csv", ABLATION_FIELDS, metric_rows)
        _atomic_csv(output_dir / "finetuning_gain.csv", GAIN_FIELDS, gain_rows)
        table_main = [
            {
                "Model": display_name(row["model"]),
                "SAR Representation": row["representation"],
                "Accuracy": row["accuracy"],
                "Balanced Accuracy": row["balanced_accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "Invalid Predictions": row["invalid_predictions"],
                "Invalid Prediction Rate": row["invalid_prediction_rate"],
                "ROC-AUC": row["roc_auc"],
                "F1 CI Low": row["f1_ci_low"],
                "F1 CI High": row["f1_ci_high"],
                "Balanced Accuracy CI Low": row["balanced_accuracy_ci_low"],
                "Balanced Accuracy CI High": row["balanced_accuracy_ci_high"],
            }
            for row in main_rows
        ]
        _atomic_csv(
            output_dir / "table_main_results.csv", MAIN_TABLE_FIELDS, table_main
        )
        table_ablation = []
        for representation in REPRESENTATIONS:
            zero = metric_index[("zero-shot", representation, "strict")]
            fine = metric_index[("fine-tuned", representation, "strict")]
            table_ablation.append(
                {
                    "Representation": representation,
                    "Zero-Shot F1": zero["f1"],
                    "Fine-Tuned F1": fine["f1"],
                    "F1 Gain": float(fine["f1"]) - float(zero["f1"]),
                    "Zero-Shot Balanced Accuracy": zero["balanced_accuracy"],
                    "Fine-Tuned Balanced Accuracy": fine["balanced_accuracy"],
                }
            )
        _atomic_csv(
            output_dir / "table_ablation.csv",
            ABLATION_TABLE_FIELDS,
            table_ablation,
        )
        consolidated = _consolidated_rows(base_predictions, ablation)
        condition_count = len(base_predictions) + len(ablation)
        expected_consolidated = len(splits["test"]) * condition_count
        if len(consolidated) != expected_consolidated:
            raise FinalResultsError("contagem incorreta de predicoes consolidadas")
        _atomic_csv(
            output_dir / "all_test_predictions.csv",
            ALL_PREDICTION_FIELDS,
            consolidated,
        )

        _save_confusion_matrices(output_dir, base_predictions, ablation)
        figures_dir = output_dir / "figures"
        _model_f1_figure(main_rows, figures_dir / "model_f1_comparison.png")
        _representation_figure(
            metric_index, figures_dir / "representation_f1_comparison.png"
        )
        _gain_figure(gain_rows, figures_dir / "finetuning_gain.png")
        _main_metrics_figure(
            main_rows, figures_dir / "model_metrics_comparison.png"
        )

        rf_metadata = _read_json(
            args.models_dir.resolve() / "random_forest_features.json"
        )
        mllm_environment = _read_json(
            args.results_dir.resolve() / "mllm_environment.json"
        )
        ablation_manifest = _read_json(
            args.results_dir.resolve() / "ablation/run_manifest.json"
        )
        split_hashes = {name: _sha256(getattr(args, name)) for name in splits}
        if (ablation_manifest.get("status") != "complete"
                or ablation_manifest.get("split_sha256") != split_hashes
                or not ablation_manifest.get("only_representation_changes")):
            raise FinalResultsError("manifesto da ablation incompleto ou splits divergentes")
        regimes = training_regimes(rf_metadata, configs["pseudo_rgb"], cnn_config,
                                   len(splits["train"]))
        regime_rows = [{
            "Model": display_name(model), "Pretraining": regime["pretraining"],
            "Total Parameters": regime["total_parameters"],
            "Trainable Parameters": regime["trainable_parameters"],
            "Training Samples": regime["training_samples"], "Epochs": regime["epochs"],
            "Batch Size": regime["batch_size"],
            "Gradient Accumulation": regime["gradient_accumulation_steps"],
            "Training Seconds": regime["training_seconds"],
            "Selection": regime["selection"], "Budget Notes": regime["budget_notes"],
        } for model, regime in regimes.items()]
        _atomic_csv(output_dir / "table_training_regimes.csv", list(regime_rows[0]), regime_rows)
        best_representation = max(
            REPRESENTATIONS,
            key=lambda rep: (
                float(metric_index[("fine-tuned", rep, "strict")]["f1"]),
                float(
                    metric_index[("fine-tuned", rep, "strict")][
                        "balanced_accuracy"
                    ]
                ),
                float(metric_index[("fine-tuned", rep, "strict")]["accuracy"]),
            ),
        )
        manifest = {
            "dataset": "Sen1Floods11 HandLabeled binary chip classification",
            "train_samples": len(splits["train"]),
            "validation_samples": len(splits["validation"]),
            "test_samples": len(splits["test"]),
            "train_events": sorted({row.event for row in splits["train"]}),
            "validation_events": sorted(
                {row.event for row in splits["validation"]}
            ),
            "test_events": sorted({row.event for row in splits["test"]}),
            "models": list(main_model_names(args.include_cnn)),
            "model_display_names": {model: DISPLAY_NAMES[model] for model in regimes},
            "main_representation": MAIN_REPRESENTATION,
            "main_representation_selection": "fixed independently of test results",
            "positive_class": "WATER",
            "representations": list(REPRESENTATIONS),
            "best_finetuned_representation_by_test_f1": best_representation,
            "best_finetuned_representation_is_exploratory": True,
            "best_finetuned_representation_notes":
                "post hoc descriptive statistic; never used to select the main models/representation",
            "training_regimes": regimes,
            "training_regime_notes":
                "different pretraining, adaptation and compute budgets; not equivalent regimes; "
                "null denotes a quantity not applicable or not recorded",
            "random_seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_interval": "95% percentile bootstrap on test samples",
            "statistical_limitations": {
                "test_events": len({row.event for row in splits["test"]}),
                "bootstrap_unit": "image (not geographic event)",
                "interpretation": "point differences and per-model image intervals alone "
                                  "do not demonstrate statistical superiority",
            },
            "evaluation_views": {
                "valid-only": "metrics only on parseable predictions",
                "strict": "invalid predictions count as classification errors",
            },
            "random_forest_configuration": rf_metadata,
            "mllm_base_model": configs["pseudo_rgb"]["base_model"],
            "mllm_revision": configs["pseudo_rgb"].get("requested_revision"),
            "lora_configuration_by_representation": configs,
            "cnn_configuration": cnn_config,
            "prompt_sha256": hashlib.sha256(
                CLASSIFICATION_PROMPT.encode("utf-8")
            ).hexdigest(),
            "software_versions": _package_versions(),
            "hardware": mllm_environment.get("hardware"),
            "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split_sha256": split_hashes,
            "ablation_execution": ablation_manifest,
            "scientific_validations": {
                "same_test_ids_all_conditions": True,
                "same_true_labels_all_conditions": True,
                "no_test_ids_in_training_jsonl": True,
                "no_test_events_in_train": True,
                "test_samples_used_for_training": 0,
                "forbidden_mllm_inputs": [],
                "only_sar_representation_changes_in_ablation": True,
                "prediction_rows": len(consolidated),
                "consolidated_conditions": condition_count,
                "no_duplicate_condition_sample_rows": True,
                "main_mllm_matches_existing_ablation": True,
                "cnn_ablation_executed": False,
            },
            "platform": platform.platform(),
        }
        _atomic_json(output_dir / "experiment_manifest.json", manifest)
        comparison_validation.update({
            "main_models": list(main_model_names(args.include_cnn)),
            "main_representation": MAIN_REPRESENTATION,
            "main_metrics_match_source_predictions": True,
            "main_mllm_matches_existing_ablation": True,
            "test_ids_equal_across_all_conditions": True,
            "true_labels_equal_across_all_conditions": True,
            "consolidated_conditions": condition_count,
            "consolidated_prediction_rows": len(consolidated),
            "no_duplicate_condition_sample_rows": True,
            "cnn_ablation_executed": False,
        })
        _atomic_json(output_dir / "comparison_validation.json", comparison_validation)
        notes = (
            "# Comparacao final\n\n"
            f"Modelos: {', '.join(display_name(model) for model in regimes)}.\n\n"
            "A representacao principal e fixada em `pseudo_rgb` para os modelos de imagem; "
            "RF usa `vv_vh_features`. Classe positiva: `WATER`. A tabela principal usa "
            "a visao `strict`; `model_comparison.csv` preserva tambem `valid-only`. "
            "`MLLM Fine-Tuned` e `Fine-Tuned` nas tabelas de ablation significam **MLLM + LoRA**. "
            "Os nomes internos anteriores permanecem nos CSVs de avaliacao.\n\n"
            "ROC-AUC dos MLLMs e indisponivel (`NaN`), pois suas respostas nao incluem "
            "probabilidades legitimas. Nenhuma probabilidade foi inferida de rotulos textuais.\n\n"
            f"O teste contem {len(splits['test'])} imagens de "
            f"{len({row.event for row in splits['test']})} eventos geograficos "
            f"({', '.join(sorted({row.event for row in splits['test']}))}). "
            "Os ICs de 95% sao bootstrap por imagem; nao representam incerteza entre eventos. "
            "Diferencas pontuais e esses ICs, isoladamente, nao demonstram superioridade "
            "estatistica entre modelos.\n\n"
            "`table_training_regimes.csv` e o manifesto documentam pre-treinamento, parametros "
            "e orcamentos diferentes. Campos vazios/nulos significam nao aplicavel ou nao "
            "registrado. Nao se assume igualdade de dados externos, tempo ou adaptacao. "
            "O total de parametros do MLLM com LoRA inclui os adapters; o total exato do "
            "zero-shot nao foi registrado separadamente.\n\n"
            f"O consolidado tem {condition_count} condicoes e {len(consolidated)} linhas. "
            "A ablation dos MLLMs em VV/VH/pseudo-RGB e reutilizada; suas condicoes pseudo-RGB "
            "nao sao duplicadas. Nenhum modelo foi retreinado e nenhuma ablation da CNN "
            "foi executada.\n\n"
            "`best_finetuned_representation_by_test_f1` e apenas descricao exploratoria "
            "posterior ao teste, sem efeito sobre a comparacao principal.\n"
        )
        (output_dir / "README.md").write_text(notes, encoding="utf-8")
    except Exception as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    print("\nResultados finais gerados")
    print(f"Modelos principais: {len(main_rows)}")
    print(f"Condicoes consolidadas: {condition_count}")
    print(f"Predicoes consolidadas: {len(consolidated)}")
    print(f"Descricao exploratoria posterior: maior F1 LoRA por representacao = {best_representation}")
    print(f"Saida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
