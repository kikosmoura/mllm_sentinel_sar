"""Inferencia deterministica e metricas compartilhadas com RF/MLLM."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from cnn.data import CNNError, SARExample
from cnn.runtime import write_csv
from evaluation.metrics import (
    ID_TO_CLASS, Prediction, calculate_metrics, load_predictions,
)


@torch.inference_mode()
def infer(model, loader, examples: list[SARExample], device, representation: str,
          criterion=None) -> tuple[list[Prediction], float | None]:
    model.eval()
    by_index = {}
    loss_sum = weight_sum = 0.0
    for images, targets, indices in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        if logits.shape != (len(targets), 2) or not bool(torch.isfinite(logits).all()):
            raise CNNError("logits invalidos/nao finitos")
        if criterion is not None:
            losses = criterion(logits, targets)
            if not bool(torch.isfinite(losses).all()):
                raise CNNError("loss de validacao nao finita")
            loss_sum += float(losses.sum())
            weight_sum += float(criterion.weight[targets].sum())
        # Softmax em float64 e p0=1-p1 atendem a tolerancia de 1e-9 do avaliador.
        probabilities = torch.softmax(logits.double(), dim=1).cpu()
        predicted = logits.argmax(dim=1).cpu().tolist()
        for offset, index in enumerate(indices.tolist()):
            if index in by_index or not 0 <= index < len(examples):
                raise CNNError("indice de predicao duplicado/fora do split")
            truth = examples[index].truth
            if int(targets[offset]) != truth.true_class_id:
                raise CNNError("rotulo de inferencia divergente do split")
            p1 = float(probabilities[offset, 1])
            if not math.isfinite(p1) or not 0 <= p1 <= 1:
                raise CNNError("probabilidade invalida")
            by_index[index] = Prediction(
                image_id=truth.image_id, event=truth.event,
                true_class=truth.true_class, true_class_id=truth.true_class_id,
                model="ResNet-18", training="supervised-transfer-learning",
                representation=representation,
                predicted_class=ID_TO_CLASS[predicted[offset]],
                prediction_valid=True, raw_response="",
                probability_non_water=1.0 - p1, probability_water=p1,
            )
    if set(by_index) != set(range(len(examples))):
        raise CNNError("predicoes nao cobrem integralmente o split")
    return [by_index[i] for i in range(len(examples))], (
        loss_sum / weight_sum if criterion is not None else None
    )


def export_predictions(path: Path, predictions: list[Prediction], examples,
                       representation: str) -> list[Prediction]:
    rows = [{
        "image_id": p.image_id, "event": p.event, "representation": p.representation,
        "true_class": p.true_class, "true_class_id": p.true_class_id,
        "predicted_class": p.predicted_class, "prediction_valid": p.prediction_valid,
        "prob_non_water": p.probability_non_water, "prob_water": p.probability_water,
    } for p in predictions]
    write_csv(path, rows)
    # Ler o arquivo realmente exportado pelo mesmo validador da comparacao final.
    return load_predictions(path, [item.truth for item in examples], model="ResNet-18",
                            training="supervised-transfer-learning",
                            representation=representation)


def metric_rows(predictions: list[Prediction], split: str, bootstrap_samples: int,
                seed: int) -> list[dict]:
    return [{"model": "ResNet-18", "split": split,
             "training": "supervised-transfer-learning",
             "representation": predictions[0].representation,
             **calculate_metrics(predictions, view, bootstrap_samples=bootstrap_samples,
                                 seed=seed).to_dict()}
            for view in ("valid-only", "strict")]
