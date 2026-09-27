"""Metadados e apresentacao da comparacao, sem carregar ou treinar modelos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from evaluation.metrics import CLASS_TO_ID, EvaluationError


LEGACY_MODELS = ("Random Forest", "MLLM Zero-Shot", "MLLM Fine-Tuned")
CNN_MODEL = "ResNet-18"
MAIN_REPRESENTATION = "pseudo_rgb"
DISPLAY_NAMES = {
    "Random Forest": "Random Forest",
    CNN_MODEL: "CNN (ResNet-18)",
    "MLLM Zero-Shot": "MLLM zero-shot",
    "MLLM Fine-Tuned": "MLLM + LoRA",
}


def main_model_names(include_cnn: bool = False) -> tuple[str, ...]:
    return (LEGACY_MODELS[0], CNN_MODEL, *LEGACY_MODELS[1:]) if include_cnn else LEGACY_MODELS


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"objeto JSON esperado: {path}")
    return value


def validate_cnn_artifacts(config_path: Path, results_dir: Path,
                           split_paths: Mapping[str, Path]) -> tuple[dict, dict]:
    """Verifica a proveniencia salva; nenhum checkpoint e desserializado."""
    config = _read_json(config_path)
    if config.get("status") != "complete" or config.get("model") != CNN_MODEL:
        raise EvaluationError("treinamento ResNet-18 incompleto ou modelo divergente")
    for key in ("test_samples_used_for_training", "test_samples_used_for_selection"):
        if config.get(key) != 0:
            raise EvaluationError(f"CNN: {key} deve ser zero")
    if config.get("representation") != MAIN_REPRESENTATION:
        raise EvaluationError("CNN principal deve usar pseudo_rgb")
    if config.get("class_to_id") != CLASS_TO_ID:
        raise EvaluationError("CNN: mapeamento de classes divergente")
    if config.get("input_source") != "SAR PNG pixels only":
        raise EvaluationError("CNN: fonte de entrada SAR nao comprovada")
    if config.get("weights_origin", {}).get("enum") != "ResNet18_Weights.IMAGENET1K_V1":
        raise EvaluationError("CNN: origem ImageNet dos pesos nao comprovada")
    hashes = {name: sha256(path) for name, path in split_paths.items()}
    if set(hashes) != {"train", "validation", "test"} or config.get("split_sha256") != hashes:
        raise EvaluationError("CNN: hashes dos splits divergem do treinamento")
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("resize") != "full-chip" or preprocessing.get("size") != [224, 224]:
        raise EvaluationError("CNN: pre-processamento do chip inteiro nao comprovado")
    phases = config.get("parameter_counts_by_phase", {})
    if config.get("best_phase") not in phases:
        raise EvaluationError("CNN: fase do checkpoint selecionado ausente")
    for counts in phases.values():
        if not 0 < counts["trainable_parameters"] <= counts["total_parameters"]:
            raise EvaluationError("CNN: contagem de parametros invalida")
    # A copia junto da configuracao permite mover a pasta do experimento.
    filename = Path(config["best_checkpoint"]).name
    checkpoint = config_path.resolve().parent / filename
    if not checkpoint.is_file() or sha256(checkpoint) != config.get("checkpoint_sha256"):
        raise EvaluationError("CNN: hash do checkpoint selecionado divergente")
    evaluation_path = results_dir / "evaluation_manifest.json"
    evaluation = _read_json(evaluation_path)
    if (evaluation.get("status") != "complete"
            or evaluation.get("checkpoint_sha256") != config["checkpoint_sha256"]
            or evaluation.get("split_sha256") != hashes
            or evaluation.get("representation") != MAIN_REPRESENTATION
            or evaluation.get("preprocessing") != preprocessing
            or evaluation.get("seed") != config.get("seed")
            or evaluation.get("test_samples_used_for_training") != 0
            or evaluation.get("test_samples_used_for_selection") != 0):
        raise EvaluationError("CNN: avaliacao nao corresponde ao checkpoint/configuracao")
    validation = {
        "cnn_configuration_sha256": sha256(config_path),
        "cnn_checkpoint_sha256": config["checkpoint_sha256"],
        "cnn_evaluation_manifest_sha256": sha256(evaluation_path),
        "cnn_test_predictions_sha256": sha256(results_dir / "test_predictions.csv"),
        "cnn_split_hashes_match": True,
        "cnn_representation": MAIN_REPRESENTATION,
        "test_samples_used_by_cnn": 0,
        "test_samples_used_for_cnn_selection": 0,
    }
    return config, validation


def training_regimes(rf: dict, lora: dict, cnn: dict | None,
                     training_samples: int) -> dict[str, dict]:
    """Distingue os orcamentos registrados; valores nao registrados ficam nulos."""
    base_pretraining = f"external pretrained multimodal model: {lora['base_model']}"
    result = {
        "Random Forest": {
            "pretraining": "none",
            "total_parameters": None, "trainable_parameters": None,
            "training_samples": training_samples, "epochs": None,
            "batch_size": None, "gradient_accumulation_steps": None,
            "training_seconds": None, "selection": "fixed hyperparameters",
            "budget_notes": "tree ensemble; neural parameter counts and epochs not applicable; "
                            "training duration not recorded",
            "n_estimators": rf["model"]["n_estimators"],
            "feature_count": rf["feature_count"],
        },
        "MLLM Zero-Shot": {
            "pretraining": base_pretraining,
            "total_parameters": None, "trainable_parameters": 0,
            "training_samples": 0, "epochs": 0,
            "batch_size": None, "gradient_accumulation_steps": None,
            "training_seconds": 0, "selection": "fixed prompt and pseudo_rgb",
            "budget_notes": "no local task training; exact zero-shot parameter count "
                            "not separately recorded; LoRA total includes adapters",
        },
        "MLLM Fine-Tuned": {
            "pretraining": base_pretraining,
            "total_parameters": lora["total_parameters"],
            "trainable_parameters": lora["trainable_parameters"],
            "training_samples": lora["training_samples"], "epochs": lora["epochs"],
            "batch_size": lora["batch_size"],
            "gradient_accumulation_steps": lora["gradient_accumulation_steps"],
            "training_seconds": lora["training_seconds"],
            "selection": "validation loss",
            "budget_notes": f"LoRA r={lora['lora_r']}, alpha={lora['lora_alpha']}; "
                            "base frozen; external pretraining budget not recorded",
            "learning_rate": lora["learning_rate"],
        },
    }
    if cnn is not None:
        counts = cnn["parameter_counts_by_phase"][cnn["best_phase"]]
        result[CNN_MODEL] = {
            "pretraining": "ImageNet-1K; ResNet18_Weights.IMAGENET1K_V1",
            **counts,
            "trainable_parameters_by_phase": cnn["parameter_counts_by_phase"],
            "training_samples": cnn["training_samples"], "epochs": cnn["epochs_executed"],
            "max_epochs": cnn["max_epochs"], "warmup_epochs": cnn["warmup_epochs"],
            "best_epoch": cnn["best_epoch"], "best_phase": cnn["best_phase"],
            "batch_size": cnn["batch_size"], "gradient_accumulation_steps": 1,
            "training_seconds": cnn["training_seconds"],
            "selection": cnn["selection"],
            "budget_notes": f"{cnn['warmup_epochs']} warmup epochs; then layer4+fc; "
                            f"max {cnn['max_epochs']} epochs, patience {cnn['patience']}; "
                            "BatchNorm statistics frozen",
            "lr_fc": cnn["lr_fc"], "lr_layer4": cnn["lr_layer4"],
        }
    return {model: result[model] for model in main_model_names(cnn is not None)}
