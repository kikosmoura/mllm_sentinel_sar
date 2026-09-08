#!/usr/bin/env python3
"""Treina e avalia o baseline Random Forest sobre features SAR por chip.

Somente as bandas 1 (VV) e 2 (VH) dos GeoTIFFs Sentinel-1 sao usadas para
calcular as features. Nenhum valor de mascara ou de porcentagem de agua entra
na matriz do modelo. NoData, NaN e infinitos sao ignorados.

Features ausentes (por exemplo, em um chip integralmente NoData) sao imputadas
com a mediana calculada exclusivamente no conjunto de treino. As mesmas
medianas sao aplicadas a validation e test e persistidas junto aos metadados
necessarios para inferencia futura.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/artigo_llm_matplotlib")
    import joblib
    import matplotlib
    import numpy as np
    import rasterio
    import sklearn

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover - depende do ambiente
    raise SystemExit(
        "Dependencia ausente. Instale 'numpy', 'rasterio', 'scikit-learn', "
        f"'matplotlib' e 'joblib' antes de executar este script ({exc})."
    ) from exc


SPLITS = ("train", "validation", "test")
CLASS_TO_ID = {"NON_WATER": 0, "WATER": 1}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}
REQUIRED_COLUMNS = {"image_id", "event", "class", "class_id", "s1_path"}
BASE_STATISTICS = ("mean", "std", "min", "max", "median", "p10", "p25", "p75", "p90")
RELATION_STATISTICS = ("mean", "std", "median")
FEATURE_NAMES = tuple(
    [f"vv_{statistic}" for statistic in BASE_STATISTICS]
    + [f"vh_{statistic}" for statistic in BASE_STATISTICS]
    + [f"vv_minus_vh_{statistic}" for statistic in RELATION_STATISTICS]
    + [f"vv_div_vh_{statistic}" for statistic in RELATION_STATISTICS]
)
FEATURE_OUTPUT_FIELDS = ["image_id", "event", "class", "class_id", "split"] + list(
    FEATURE_NAMES
)
PREDICTION_FIELDS = [
    "image_id",
    "event",
    "true_class",
    "true_class_id",
    "predicted_class",
    "predicted_class_id",
    "prob_non_water",
    "prob_water",
]
METRIC_FIELDS = [
    "model",
    "split",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "samples",
]


class BaselineError(ValueError):
    """Erro de consistencia que impede um baseline confiavel."""


@dataclass(frozen=True)
class SplitData:
    name: str
    path: Path
    rows: list[dict[str, str]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrai features estatisticas de VV/VH, treina um Random Forest "
            "somente em train e avalia separadamente validation e test."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="CSV do split geografico de treino.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
        help="CSV do split geografico de validacao.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/splits/test.csv"),
        help="CSV do split geografico de teste.",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=Path("data/features/random_forest_features.csv"),
        help="CSV tabular de features imputadas.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/random_forest"),
        help="Diretorio das metricas, predicoes e figuras.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/random_forest.joblib"),
        help="Arquivo joblib do RandomForestClassifier treinado.",
    )
    parser.add_argument(
        "--feature-metadata-output",
        type=Path,
        default=None,
        help=(
            "JSON com ordem das features e medianas de imputacao. Por padrao, "
            "usa <model-output sem extensao>_features.json."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente do Random Forest.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Numero de arvores do Random Forest.",
    )
    args = parser.parse_args(argv)

    for argument_name in ("train", "validation", "test"):
        path = getattr(args, argument_name)
        if not path.is_file():
            parser.error(f"--{argument_name} nao e um arquivo acessivel: {path}")
    if args.n_estimators <= 0:
        parser.error("--n-estimators deve ser maior que zero")
    if args.feature_metadata_output is None:
        args.feature_metadata_output = args.model_output.with_name(
            f"{args.model_output.stem}_features.json"
        )
    return args


def load_split(name: str, path: Path) -> SplitData:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise BaselineError(f"{path}: CSV sem cabecalho")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise BaselineError(
                f"{path}: coluna(s) ausente(s): {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise BaselineError(f"split {name} esta vazio")
    return SplitData(name=name, path=path.resolve(), rows=rows)


def resolve_s1_path(raw_path: str, split_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = split_path.parent / path
    return path.resolve()


def validate_splits(split_data: dict[str, SplitData]) -> None:
    id_sets: dict[str, set[str]] = {}
    event_sets: dict[str, set[str]] = {}
    seen_s1_paths: dict[Path, str] = {}

    for split_name in SPLITS:
        data = split_data[split_name]
        ids: set[str] = set()
        events: set[str] = set()
        classes: set[int] = set()
        for row_number, row in enumerate(data.rows, start=2):
            image_id = row.get("image_id", "").strip()
            event = row.get("event", "").strip()
            chip_class = row.get("class", "").strip()
            if not image_id or not event:
                raise BaselineError(
                    f"{data.path}:{row_number}: image_id/event vazio"
                )
            image_key = image_id.casefold()
            if image_key in ids:
                raise BaselineError(
                    f"image_id duplicado em {split_name}: {image_id!r}"
                )
            ids.add(image_key)
            if chip_class not in CLASS_TO_ID:
                raise BaselineError(
                    f"classe invalida para {image_id!r}: {chip_class!r}"
                )
            try:
                class_id = int(row.get("class_id", ""))
            except ValueError as exc:
                raise BaselineError(
                    f"class_id invalido para {image_id!r}: {row.get('class_id')!r}"
                ) from exc
            if class_id != CLASS_TO_ID[chip_class]:
                raise BaselineError(
                    f"classe/class_id inconsistentes para {image_id!r}: "
                    f"{chip_class}/{class_id}"
                )
            classes.add(class_id)
            if "split" in row and row["split"].strip() != split_name:
                raise BaselineError(
                    f"split declarado incorretamente para {image_id!r}: "
                    f"{row['split']!r}, esperado {split_name!r}"
                )
            inferred_event, separator, chip_component = image_id.rpartition("_")
            if not separator or not chip_component or inferred_event != event:
                raise BaselineError(
                    f"evento inconsistente para {image_id!r}: {event!r}"
                )
            events.add(event)

            s1_path = resolve_s1_path(row.get("s1_path", ""), data.path)
            if not s1_path.is_file():
                raise BaselineError(
                    f"GeoTIFF S1 inexistente para {image_id!r}: {s1_path}"
                )
            previous_id = seen_s1_paths.get(s1_path)
            if previous_id is not None and previous_id.casefold() != image_key:
                raise BaselineError(
                    f"GeoTIFF S1 reutilizado por IDs distintos: "
                    f"{previous_id!r} e {image_id!r}"
                )
            seen_s1_paths[s1_path] = image_id

        if classes != {0, 1}:
            raise BaselineError(
                f"split {split_name} nao contem as duas classes: {sorted(classes)}"
            )
        id_sets[split_name] = ids
        event_sets[split_name] = events

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlapping_ids = id_sets[left] & id_sets[right]
        if overlapping_ids:
            raise BaselineError(
                f"image_id em mais de um split ({left}/{right}): "
                f"{sorted(overlapping_ids)}"
            )
        overlapping_events = event_sets[left] & event_sets[right]
        if overlapping_events:
            raise BaselineError(
                f"evento em mais de um split ({left}/{right}): "
                f"{sorted(overlapping_events)}"
            )


def finite_values(masked_band: np.ma.MaskedArray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(masked_band.data, dtype=np.float64)
    valid = ~np.ma.getmaskarray(masked_band) & np.isfinite(values)
    return values, valid


def base_statistics(values: np.ndarray, valid: np.ndarray) -> list[float]:
    selected = values[valid]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return [math.nan] * len(BASE_STATISTICS)
    percentiles = np.percentile(selected, [10.0, 25.0, 50.0, 75.0, 90.0])
    return [
        float(np.mean(selected)),
        float(np.std(selected, ddof=0)),
        float(np.min(selected)),
        float(np.max(selected)),
        float(percentiles[2]),
        float(percentiles[0]),
        float(percentiles[1]),
        float(percentiles[3]),
        float(percentiles[4]),
    ]


def relation_statistics(values: np.ndarray, valid: np.ndarray) -> list[float]:
    selected = values[valid]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return [math.nan] * len(RELATION_STATISTICS)
    return [
        float(np.mean(selected)),
        float(np.std(selected, ddof=0)),
        float(np.median(selected)),
    ]


def extract_sar_features(s1_path: Path) -> np.ndarray:
    try:
        with rasterio.open(s1_path) as dataset:
            if dataset.count < 2:
                raise BaselineError(
                    f"GeoTIFF possui {dataset.count} banda(s); VV e VH sao obrigatorias"
                )
            bands = dataset.read([1, 2], masked=True)
    except BaselineError:
        raise
    except Exception as exc:
        raise BaselineError(f"nao foi possivel ler {s1_path}: {exc}") from exc

    vv, vv_valid = finite_values(bands[0])
    vh, vh_valid = finite_values(bands[1])
    common_valid = vv_valid & vh_valid

    difference = np.full(vv.shape, math.nan, dtype=np.float64)
    np.subtract(vv, vh, out=difference, where=common_valid)

    ratio_valid = common_valid & (vh != 0.0)
    ratio = np.full(vv.shape, math.nan, dtype=np.float64)
    np.divide(vv, vh, out=ratio, where=ratio_valid)
    ratio_valid &= np.isfinite(ratio)

    features = np.asarray(
        base_statistics(vv, vv_valid)
        + base_statistics(vh, vh_valid)
        + relation_statistics(difference, common_valid)
        + relation_statistics(ratio, ratio_valid),
        dtype=np.float64,
    )
    if features.shape != (len(FEATURE_NAMES),):
        raise BaselineError(
            f"vetor de features com forma {features.shape}; "
            f"esperado {(len(FEATURE_NAMES),)}"
        )
    return features


def extract_all_features(
    split_data: dict[str, SplitData],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, str]]]]:
    matrices: dict[str, np.ndarray] = {}
    normalized_rows: dict[str, list[dict[str, str]]] = {}
    errors: list[str] = []
    for split_name in SPLITS:
        feature_rows: list[np.ndarray] = []
        normalized_rows[split_name] = []
        for row in split_data[split_name].rows:
            image_id = row["image_id"].strip()
            s1_path = resolve_s1_path(row["s1_path"], split_data[split_name].path)
            try:
                feature_rows.append(extract_sar_features(s1_path))
                normalized_row = dict(row)
                normalized_row["s1_path"] = str(s1_path)
                normalized_rows[split_name].append(normalized_row)
            except Exception as exc:
                errors.append(f"{split_name}/{image_id}: {exc}")
        if feature_rows:
            matrices[split_name] = np.vstack(feature_rows)
        else:
            matrices[split_name] = np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if errors:
        raise BaselineError("falhas na extracao:\n  - " + "\n  - ".join(errors))
    return matrices, normalized_rows


def impute_from_train(
    raw_matrices: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    train_matrix = raw_matrices["train"]
    medians = np.empty(len(FEATURE_NAMES), dtype=np.float64)
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        finite_train = train_matrix[
            np.isfinite(train_matrix[:, feature_index]), feature_index
        ]
        if finite_train.size == 0:
            raise BaselineError(
                f"feature {feature_name!r} nao possui valores finitos no treino"
            )
        medians[feature_index] = np.median(finite_train)

    imputed: dict[str, np.ndarray] = {}
    imputed_counts: dict[str, int] = {}
    for split_name in SPLITS:
        matrix = raw_matrices[split_name].copy()
        missing = ~np.isfinite(matrix)
        imputed_counts[split_name] = int(np.count_nonzero(missing))
        if bool(np.any(missing)):
            row_indices, column_indices = np.where(missing)
            matrix[row_indices, column_indices] = medians[column_indices]
        if not bool(np.all(np.isfinite(matrix))):
            raise BaselineError(
                f"features nao finitas permaneceram apos imputacao em {split_name}"
            )
        imputed[split_name] = matrix
    return imputed, medians, imputed_counts


def atomic_write_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(output_path: Path, payload: object) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_save_figure(figure: plt.Figure, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        figure.savefig(temporary_path, format="png", dpi=150, bbox_inches="tight")
        temporary_path.replace(output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        plt.close(figure)


def atomic_dump_model(model: RandomForestClassifier, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        joblib.dump(model, temporary_path, compress=3)
        loaded_model = joblib.load(temporary_path)
        if not isinstance(loaded_model, RandomForestClassifier):
            raise BaselineError("artefato joblib salvo nao e um RandomForestClassifier")
        temporary_path.replace(output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def feature_output_rows(
    normalized_rows: dict[str, list[dict[str, str]]],
    matrices: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in SPLITS:
        for source_row, feature_values in zip(
            normalized_rows[split_name], matrices[split_name], strict=True
        ):
            output_row: dict[str, object] = {
                "image_id": source_row["image_id"],
                "event": source_row["event"],
                "class": source_row["class"],
                "class_id": int(source_row["class_id"]),
                "split": split_name,
            }
            output_row.update(
                {
                    feature_name: float(feature_values[index])
                    for index, feature_name in enumerate(FEATURE_NAMES)
                }
            )
            rows.append(output_row)
    return rows


def evaluate_split(
    model: RandomForestClassifier,
    split_name: str,
    matrix: np.ndarray,
    rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    y_true = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    y_pred = model.predict(matrix).astype(np.int64)
    probabilities = model.predict_proba(matrix)
    class_indices = {int(class_id): index for index, class_id in enumerate(model.classes_)}
    if set(class_indices) != {0, 1}:
        raise BaselineError(
            f"classes inesperadas no modelo: {sorted(class_indices)}"
        )
    prob_non_water = probabilities[:, class_indices[0]]
    prob_water = probabilities[:, class_indices[1]]
    if len(y_pred) != len(rows):
        raise BaselineError(
            f"quantidade de predicoes em {split_name} difere do split"
        )
    if not np.allclose(prob_non_water + prob_water, 1.0, rtol=0.0, atol=1e-12):
        raise BaselineError(
            f"probabilidades em {split_name} nao somam aproximadamente 1"
        )
    if not bool(np.all(np.isfinite(probabilities))):
        raise BaselineError(f"probabilidades nao finitas em {split_name}")

    roc_auc = (
        float(roc_auc_score(y_true, prob_water))
        if np.unique(y_true).size == 2
        else math.nan
    )
    metrics: dict[str, object] = {
        "model": "RandomForest",
        "split": split_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": roc_auc,
        "samples": len(rows),
    }
    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        predicted_id = int(y_pred[index])
        prediction_rows.append(
            {
                "image_id": row["image_id"],
                "event": row["event"],
                "true_class": row["class"],
                "true_class_id": int(row["class_id"]),
                "predicted_class": ID_TO_CLASS[predicted_id],
                "predicted_class_id": predicted_id,
                "prob_non_water": float(prob_non_water[index]),
                "prob_water": float(prob_water[index]),
            }
        )
    matrix_confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, prediction_rows, matrix_confusion


def confusion_figure(matrix: np.ndarray, split_name: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(4.5, 4.0))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["NON_WATER", "WATER"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"Random Forest - {split_name}")
    figure.tight_layout()
    return figure


def importance_figure(importance_rows: list[dict[str, object]]) -> plt.Figure:
    selected = list(reversed(importance_rows[:15]))
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.barh(
        [str(row["feature"]) for row in selected],
        [float(row["importance"]) for row in selected],
    )
    axis.set_xlabel("Importance")
    axis.set_title("Random Forest - 15 most important features")
    figure.tight_layout()
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_paths = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    try:
        split_data = {
            name: load_split(name, path) for name, path in split_paths.items()
        }
        validate_splits(split_data)
        forbidden_tokens = ("label", "mask", "percentage", "class", "target")
        if any(
            token in feature_name.casefold()
            for feature_name in FEATURE_NAMES
            for token in forbidden_tokens
        ):
            raise BaselineError("a lista de features contem informacao proibida")
        raw_matrices, normalized_rows = extract_all_features(split_data)
        matrices, imputation_medians, imputed_counts = impute_from_train(
            raw_matrices
        )
    except (OSError, csv.Error, BaselineError) as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    y_train = np.asarray(
        [int(row["class_id"]) for row in normalized_rows["train"]],
        dtype=np.int64,
    )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    # Este e o unico ponto de ajuste do modelo; somente train e utilizado.
    model.fit(matrices["train"], y_train)

    try:
        metrics_rows: list[dict[str, object]] = []
        predictions: dict[str, list[dict[str, object]]] = {}
        confusion_matrices: dict[str, np.ndarray] = {}
        for split_name in ("validation", "test"):
            metrics, split_predictions, matrix_confusion = evaluate_split(
                model,
                split_name,
                matrices[split_name],
                normalized_rows[split_name],
            )
            metrics_rows.append(metrics)
            predictions[split_name] = split_predictions
            confusion_matrices[split_name] = matrix_confusion

        if len(predictions["validation"]) != len(split_data["validation"].rows):
            raise BaselineError("numero incorreto de predicoes de validation")
        if len(predictions["test"]) != len(split_data["test"].rows):
            raise BaselineError("numero incorreto de predicoes de test")

        feature_rows = feature_output_rows(normalized_rows, matrices)
        if len(feature_rows) != sum(len(data.rows) for data in split_data.values()):
            raise BaselineError("numero incorreto de linhas no CSV de features")
        feature_matrix = np.asarray(
            [
                [float(row[feature_name]) for feature_name in FEATURE_NAMES]
                for row in feature_rows
            ],
            dtype=np.float64,
        )
        if not bool(np.all(np.isfinite(feature_matrix))):
            raise BaselineError("CSV de features conteria valor nao finito")

        importance_rows = [
            {"feature": feature_name, "importance": float(importance)}
            for feature_name, importance in zip(
                FEATURE_NAMES, model.feature_importances_, strict=True
            )
        ]
        importance_rows.sort(
            key=lambda row: (-float(row["importance"]), str(row["feature"]))
        )
        if not math.isclose(
            sum(float(row["importance"]) for row in importance_rows),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BaselineError("importancias das features nao somam 1")

        results_dir = args.results_dir.resolve()
        atomic_write_csv(args.features_output, FEATURE_OUTPUT_FIELDS, feature_rows)
        atomic_write_csv(
            results_dir / "validation_predictions.csv",
            PREDICTION_FIELDS,
            predictions["validation"],
        )
        atomic_write_csv(
            results_dir / "test_predictions.csv",
            PREDICTION_FIELDS,
            predictions["test"],
        )
        atomic_write_csv(results_dir / "metrics.csv", METRIC_FIELDS, metrics_rows)
        atomic_write_csv(
            results_dir / "feature_importance.csv",
            ["feature", "importance"],
            importance_rows,
        )
        for split_name in ("validation", "test"):
            atomic_save_figure(
                confusion_figure(confusion_matrices[split_name], split_name),
                results_dir / f"confusion_matrix_{split_name}.png",
            )
        atomic_save_figure(
            importance_figure(importance_rows),
            results_dir / "feature_importance.png",
        )
        atomic_dump_model(model, args.model_output)
        metadata = {
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "feature_source": "Sentinel-1 GeoTIFF bands only",
            "bands": {"1": "VV", "2": "VH"},
            "statistics": {
                "vv_vh": list(BASE_STATISTICS),
                "vv_minus_vh": list(RELATION_STATISTICS),
                "vv_div_vh": list(RELATION_STATISTICS),
                "standard_deviation_ddof": 0,
            },
            "invalid_pixels": "NoData, NaN and infinity ignored; division by zero ignored",
            "imputation": {
                "strategy": "per-feature median fitted on train only",
                "values_in_feature_order": [
                    float(value) for value in imputation_medians
                ],
                "values_by_feature": {
                    feature_name: float(imputation_medians[index])
                    for index, feature_name in enumerate(FEATURE_NAMES)
                },
                "imputed_cells_by_split": imputed_counts,
            },
            "model": {
                "type": "RandomForestClassifier",
                "n_estimators": args.n_estimators,
                "random_state": args.random_state,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
            "versions": {
                "numpy": np.__version__,
                "rasterio": rasterio.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
                "matplotlib": matplotlib.__version__,
            },
        }
        atomic_write_json(args.feature_metadata_output, metadata)
    except Exception as exc:
        print(f"[ERRO FATAL] Falha ao gerar artefatos: {exc}", file=sys.stderr)
        return 1

    print("\nRelatorio Random Forest")
    for split_name in SPLITS:
        class_counts = Counter(row["class"] for row in normalized_rows[split_name])
        print(
            f"{split_name}: {len(normalized_rows[split_name])} exemplos "
            f"(NON_WATER={class_counts['NON_WATER']}, WATER={class_counts['WATER']})"
        )
    print(f"Features SAR: {len(FEATURE_NAMES)}")
    print(
        "Celulas imputadas com medianas do treino: "
        + ", ".join(f"{name}={imputed_counts[name]}" for name in SPLITS)
    )
    print("\nMetricas")
    for metrics in metrics_rows:
        print(
            f"{metrics['split']}: accuracy={metrics['accuracy']:.6f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f}, "
            f"precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, f1={metrics['f1']:.6f}, "
            f"roc_auc={metrics['roc_auc']:.6f}"
        )
    print("\n10 features mais importantes")
    for row in importance_rows[:10]:
        print(f"{row['feature']}: {row['importance']:.8f}")
    print("\nValidacoes: splits exclusivos; duas classes por split; features finitas;")
    print("somente features SAR; probabilidades normalizadas; predicoes completas.")
    print(f"Modelo: {args.model_output.resolve()}")
    print(f"Metadados de features: {args.feature_metadata_output.resolve()}")
    print(f"Resultados: {args.results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
