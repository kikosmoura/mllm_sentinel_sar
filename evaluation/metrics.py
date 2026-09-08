"""Leitura, validacao e metricas uniformes para todos os classificadores."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


CLASS_TO_ID = {"NON_WATER": 0, "WATER": 1}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
EVALUATION_VIEWS = ("valid-only", "strict")


class EvaluationError(ValueError):
    """Inconsistencia que impede uma comparacao experimental confiavel."""


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    event: str
    true_class: str
    true_class_id: int


@dataclass(frozen=True)
class Prediction:
    image_id: str
    event: str
    true_class: str
    true_class_id: int
    model: str
    training: str
    representation: str
    predicted_class: str | None
    prediction_valid: bool
    raw_response: str
    probability_non_water: float | None = None
    probability_water: float | None = None


@dataclass(frozen=True)
class MetricResult:
    evaluation_view: str
    samples: int
    evaluated_samples: int
    valid_predictions: int
    invalid_predictions: int
    invalid_prediction_rate: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    f1_ci_low: float
    f1_ci_high: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float
    metric_notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise EvaluationError(f"CSV inexistente: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise EvaluationError(f"CSV sem cabecalho: {path}")
            return list(reader.fieldnames), list(reader)
    except (OSError, csv.Error) as exc:
        raise EvaluationError(f"nao foi possivel ler {path}: {exc}") from exc


def _parse_class(class_name: str, class_id_raw: str, context: str) -> int:
    class_name = class_name.strip()
    if class_name not in CLASS_TO_ID:
        raise EvaluationError(f"classe invalida em {context}: {class_name!r}")
    try:
        class_id = int(class_id_raw)
    except ValueError as exc:
        raise EvaluationError(
            f"class_id invalido em {context}: {class_id_raw!r}"
        ) from exc
    if class_id != CLASS_TO_ID[class_name]:
        raise EvaluationError(
            f"classe/class_id divergentes em {context}: {class_name}/{class_id}"
        )
    return class_id


def load_ground_truth(path: Path, expected_split: str = "test") -> list[GroundTruth]:
    fields, rows = _read_csv(path)
    required = {"image_id", "event", "class", "class_id"}
    missing = required - set(fields)
    if missing:
        raise EvaluationError(
            f"{path}: colunas ausentes: {', '.join(sorted(missing))}"
        )
    if not rows:
        raise EvaluationError(f"ground truth vazio: {path}")

    truth: list[GroundTruth] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        event = row["event"].strip()
        context = f"{path}:{line_number}"
        if not image_id or not event:
            raise EvaluationError(f"image_id/event vazio em {context}")
        key = image_id.casefold()
        if key in seen_ids:
            raise EvaluationError(f"image_id duplicado em {path}: {image_id}")
        seen_ids.add(key)
        class_id = _parse_class(row["class"], row["class_id"], context)
        if "split" in row and row["split"].strip() != expected_split:
            raise EvaluationError(
                f"split incorreto em {context}: {row['split']!r}, "
                f"esperado {expected_split!r}"
            )
        inferred_event, separator, chip = image_id.rpartition("_")
        if not separator or not chip or inferred_event != event:
            raise EvaluationError(
                f"evento {event!r} inconsistente com image_id {image_id!r}"
            )
        truth.append(
            GroundTruth(
                image_id=image_id,
                event=event,
                true_class=row["class"].strip(),
                true_class_id=class_id,
            )
        )
    if {row.true_class_id for row in truth} != {0, 1}:
        raise EvaluationError("ground truth precisa conter as duas classes")
    return truth


def _parse_boolean(value: str, context: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise EvaluationError(f"booleano invalido em {context}: {value!r}")


def _optional_probability(value: str | None, context: str) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise EvaluationError(f"probabilidade invalida em {context}: {value!r}") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise EvaluationError(f"probabilidade fora de [0,1] em {context}: {result}")
    return result


def load_predictions(
    path: Path,
    ground_truth: Sequence[GroundTruth],
    *,
    model: str,
    training: str,
    representation: str,
) -> list[Prediction]:
    fields, rows = _read_csv(path)
    required = {
        "image_id",
        "event",
        "true_class",
        "true_class_id",
        "predicted_class",
    }
    missing = required - set(fields)
    if missing:
        raise EvaluationError(
            f"{path}: colunas ausentes: {', '.join(sorted(missing))}"
        )
    truth_by_id = {row.image_id: row for row in ground_truth}
    parsed_by_id: dict[str, Prediction] = {}
    for line_number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        context = f"{path}:{line_number}"
        if image_id in parsed_by_id:
            raise EvaluationError(f"image_id duplicado em {path}: {image_id}")
        expected = truth_by_id.get(image_id)
        if expected is None:
            raise EvaluationError(f"ID fora do ground truth em {context}: {image_id!r}")
        true_class_id = _parse_class(
            row["true_class"], row["true_class_id"], context
        )
        if (
            row["event"].strip() != expected.event
            or row["true_class"].strip() != expected.true_class
            or true_class_id != expected.true_class_id
        ):
            raise EvaluationError(f"ground truth divergente em {context}")
        if "representation" in row and row["representation"].strip() != representation:
            raise EvaluationError(
                f"representacao divergente em {context}: "
                f"{row['representation']!r} != {representation!r}"
            )

        predicted_class_raw = row["predicted_class"].strip()
        if "prediction_valid" in row:
            prediction_valid = _parse_boolean(row["prediction_valid"], context)
        else:
            prediction_valid = predicted_class_raw in CLASS_TO_ID
        if prediction_valid and predicted_class_raw not in CLASS_TO_ID:
            raise EvaluationError(
                f"predicao marcada valida mas classe invalida em {context}"
            )
        if not prediction_valid and predicted_class_raw:
            raise EvaluationError(
                f"predicao invalida deveria ter predicted_class vazio em {context}"
            )
        probability_non_water = _optional_probability(
            row.get("prob_non_water") or row.get("probability_non_water"), context
        )
        probability_water = _optional_probability(
            row.get("prob_water") or row.get("probability_water"), context
        )
        if (probability_non_water is None) != (probability_water is None):
            raise EvaluationError(f"par de probabilidades incompleto em {context}")
        if probability_water is not None and not math.isclose(
            probability_non_water + probability_water,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise EvaluationError(f"probabilidades nao somam 1 em {context}")
        parsed_by_id[image_id] = Prediction(
            image_id=image_id,
            event=expected.event,
            true_class=expected.true_class,
            true_class_id=expected.true_class_id,
            model=model,
            training=training,
            representation=representation,
            predicted_class=predicted_class_raw if prediction_valid else None,
            prediction_valid=prediction_valid,
            raw_response=row.get("raw_response", ""),
            probability_non_water=probability_non_water,
            probability_water=probability_water,
        )

    expected_ids = set(truth_by_id)
    actual_ids = set(parsed_by_id)
    if actual_ids != expected_ids:
        missing_ids = sorted(expected_ids - actual_ids)
        extra_ids = sorted(actual_ids - expected_ids)
        raise EvaluationError(
            f"IDs de {path} diferem do teste; ausentes={missing_ids}, extras={extra_ids}"
        )
    return [parsed_by_id[row.image_id] for row in ground_truth]


def confusion_counts(predictions: Sequence[Prediction]) -> np.ndarray:
    """Matriz 2x2 sobre previsoes validas, na ordem NON_WATER/WATER."""
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in predictions:
        if not row.prediction_valid or row.predicted_class is None:
            continue
        matrix[row.true_class_id, CLASS_TO_ID[row.predicted_class]] += 1
    return matrix


def _point_metrics(
    predictions: Sequence[Prediction], view: Literal["valid-only", "strict"]
) -> tuple[dict[str, float], str]:
    if view not in EVALUATION_VIEWS:
        raise EvaluationError(f"visao de avaliacao invalida: {view!r}")
    selected = (
        [row for row in predictions if row.prediction_valid]
        if view == "valid-only"
        else list(predictions)
    )
    if not selected:
        empty = {
            name: math.nan
            for name in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            )
        }
        return empty, "nenhuma previsao valida para calcular metricas"

    true_water = sum(row.true_class_id == 1 for row in selected)
    true_non_water = len(selected) - true_water
    tp = tn = fp = 0
    for row in selected:
        if not row.prediction_valid or row.predicted_class is None:
            continue
        predicted_id = CLASS_TO_ID[row.predicted_class]
        if row.true_class_id == 1 and predicted_id == 1:
            tp += 1
        elif row.true_class_id == 0 and predicted_id == 0:
            tn += 1
        elif row.true_class_id == 0 and predicted_id == 1:
            fp += 1

    accuracy = (tp + tn) / len(selected)
    recall = tp / true_water if true_water else math.nan
    specificity = tn / true_non_water if true_non_water else math.nan
    balanced_accuracy = (
        (recall + specificity) / 2.0
        if math.isfinite(recall) and math.isfinite(specificity)
        else math.nan
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(recall) and precision + recall
        else 0.0
    )

    notes: list[str] = []
    probabilities_available = all(
        row.probability_water is not None and row.prediction_valid for row in selected
    )
    if probabilities_available and {row.true_class_id for row in selected} == {0, 1}:
        roc_auc = float(
            roc_auc_score(
                [row.true_class_id for row in selected],
                [float(row.probability_water) for row in selected],
            )
        )
    else:
        roc_auc = math.nan
        if not probabilities_available:
            notes.append("ROC-AUC indisponivel: probabilidades ausentes")
        else:
            notes.append("ROC-AUC indisponivel: apenas uma classe na avaliacao")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }, "; ".join(notes)


def _bootstrap_intervals(
    predictions: Sequence[Prediction],
    view: Literal["valid-only", "strict"],
    samples: int,
    seed: int,
) -> tuple[float, float, float, float, str]:
    if samples <= 0:
        return math.nan, math.nan, math.nan, math.nan, "bootstrap desativado"
    pool = (
        [row for row in predictions if row.prediction_valid]
        if view == "valid-only"
        else list(predictions)
    )
    if len(pool) < 2 or {row.true_class_id for row in pool} != {0, 1}:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "IC bootstrap indisponivel: amostra insuficiente ou uma unica classe",
        )
    rng = np.random.default_rng(seed)
    f1_values: list[float] = []
    balanced_values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(pool), size=len(pool))
        replicate = [pool[int(index)] for index in indices]
        metrics, _ = _point_metrics(replicate, view)
        if math.isfinite(metrics["f1"]):
            f1_values.append(metrics["f1"])
        if math.isfinite(metrics["balanced_accuracy"]):
            balanced_values.append(metrics["balanced_accuracy"])
    minimum_valid = math.ceil(samples * 0.9)
    if len(f1_values) < minimum_valid or len(balanced_values) < minimum_valid:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            "IC bootstrap indisponivel: menos de 90% das replicas validas",
        )
    f1_low, f1_high = np.percentile(f1_values, [2.5, 97.5])
    balanced_low, balanced_high = np.percentile(balanced_values, [2.5, 97.5])
    return (
        float(f1_low),
        float(f1_high),
        float(balanced_low),
        float(balanced_high),
        "",
    )


def calculate_metrics(
    predictions: Sequence[Prediction],
    view: Literal["valid-only", "strict"],
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> MetricResult:
    if not predictions:
        raise EvaluationError("lista de predicoes vazia")
    point, metric_note = _point_metrics(predictions, view)
    invalid = sum(not row.prediction_valid for row in predictions)
    evaluated = (
        len(predictions) - invalid if view == "valid-only" else len(predictions)
    )
    f1_low, f1_high, balanced_low, balanced_high, bootstrap_note = (
        _bootstrap_intervals(predictions, view, bootstrap_samples, seed)
    )
    notes = "; ".join(note for note in (metric_note, bootstrap_note) if note)
    return MetricResult(
        evaluation_view=view,
        samples=len(predictions),
        evaluated_samples=evaluated,
        valid_predictions=len(predictions) - invalid,
        invalid_predictions=invalid,
        invalid_prediction_rate=invalid / len(predictions),
        accuracy=point["accuracy"],
        balanced_accuracy=point["balanced_accuracy"],
        precision=point["precision"],
        recall=point["recall"],
        f1=point["f1"],
        roc_auc=point["roc_auc"],
        f1_ci_low=f1_low,
        f1_ci_high=f1_high,
        balanced_accuracy_ci_low=balanced_low,
        balanced_accuracy_ci_high=balanced_high,
        metric_notes=notes,
    )


def validate_prediction_sets(
    prediction_sets: Iterable[Sequence[Prediction]],
) -> None:
    sets = list(prediction_sets)
    if not sets:
        raise EvaluationError("nenhum conjunto de predicoes informado")
    reference = [
        (row.image_id, row.event, row.true_class, row.true_class_id) for row in sets[0]
    ]
    for index, rows in enumerate(sets[1:], start=2):
        candidate = [
            (row.image_id, row.event, row.true_class, row.true_class_id) for row in rows
        ]
        if candidate != reference:
            raise EvaluationError(
                f"conjunto de predicoes {index} usa IDs/rotulos diferentes"
            )
