#!/usr/bin/env python3
"""Cria splits deterministicas e mutuamente exclusivos por evento geografico.

Como o Sen1Floods11 possui poucos eventos, todas as atribuicoes possiveis de
eventos aos tres splits sao avaliadas. A funcao objetivo prioriza, nesta ordem:

1. minimizar splits sem uma das duas classes;
2. minimizar conjuntamente o desvio da quantidade total e das quantidades por
   classe em relacao as proporcoes solicitadas;
3. desempatar pela ordem alfabetica dos eventos.

Assim, um evento inteiro recebe exatamente um split e nunca ha leakage por
chips do mesmo evento em conjuntos diferentes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"image_id", "event", "class", "class_id"}
EXPECTED_CLASS_IDS = {"NON_WATER": 0, "WATER": 1}
SUMMARY_FIELDS = [
    "split",
    "events",
    "samples",
    "water",
    "non_water",
    "water_percentage",
]


class InvalidDatasetError(ValueError):
    """Erro que impede a construcao confiavel dos splits."""


@dataclass(frozen=True)
class EventStats:
    event: str
    samples: int
    water: int
    non_water: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Divide o dataset por evento geografico usando busca exaustiva "
            "deterministica, sem separar chips do mesmo evento."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experimental_dataset.csv"),
        help="Dataset experimental binario.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Diretorio dos CSVs de train, validation, test e resumo.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proporcao alvo de amostras para treino.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Proporcao alvo de amostras para validacao.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporcao alvo de amostras para teste.",
    )
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        parser.error(f"--dataset nao e um arquivo acessivel: {args.dataset}")
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    if not all(math.isfinite(value) and value > 0.0 for value in ratios):
        parser.error("todas as proporcoes devem ser finitas e maiores que zero")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        parser.error("--train-ratio + --validation-ratio + --test-ratio deve ser 1")
    return args


def load_and_validate_dataset(
    dataset_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    with dataset_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InvalidDatasetError("CSV sem cabecalho")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise InvalidDatasetError(
                "CSV sem coluna(s) obrigatoria(s): " + ", ".join(sorted(missing))
            )
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise InvalidDatasetError("dataset experimental vazio")
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = row.get("image_id", "").strip()
        event = row.get("event", "").strip()
        chip_class = row.get("class", "").strip()
        if not image_id:
            raise InvalidDatasetError(f"image_id vazio na linha {row_number}")
        image_key = image_id.casefold()
        if image_key in seen_ids:
            raise InvalidDatasetError(f"image_id duplicado: {image_id!r}")
        seen_ids.add(image_key)
        if chip_class not in EXPECTED_CLASS_IDS:
            raise InvalidDatasetError(
                f"classe invalida na linha {row_number}: {chip_class!r}"
            )
        try:
            class_id = int(row.get("class_id", ""))
        except ValueError as exc:
            raise InvalidDatasetError(
                f"class_id invalido na linha {row_number}: {row.get('class_id')!r}"
            ) from exc
        if class_id != EXPECTED_CLASS_IDS[chip_class]:
            raise InvalidDatasetError(
                f"classe/class_id inconsistentes para {image_id!r}: "
                f"{chip_class}/{class_id}"
            )
        inferred_event, separator, chip_component = image_id.rpartition("_")
        if not separator or not inferred_event or not chip_component:
            raise InvalidDatasetError(
                f"image_id sem evento extraivel na linha {row_number}: {image_id!r}"
            )
        if event != inferred_event:
            raise InvalidDatasetError(
                f"evento inconsistente para {image_id!r}: "
                f"CSV={event!r}, esperado={inferred_event!r}"
            )
    return rows, fieldnames


def calculate_event_stats(rows: list[dict[str, str]]) -> list[EventStats]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["event"]].append(row)
    stats = []
    for event in sorted(grouped, key=str.casefold):
        event_rows = grouped[event]
        water = sum(row["class"] == "WATER" for row in event_rows)
        non_water = sum(row["class"] == "NON_WATER" for row in event_rows)
        stats.append(EventStats(event, len(event_rows), water, non_water))
    return stats


def assignment_score(
    assignment: tuple[int, ...],
    event_stats: list[EventStats],
    ratios: tuple[float, float, float],
) -> tuple[int, float, float, tuple[int, ...]]:
    samples = [0, 0, 0]
    water = [0, 0, 0]
    non_water = [0, 0, 0]
    for event_index, split_index in enumerate(assignment):
        stats = event_stats[event_index]
        samples[split_index] += stats.samples
        water[split_index] += stats.water
        non_water[split_index] += stats.non_water

    total_samples = sum(samples)
    total_water = sum(water)
    total_non_water = sum(non_water)
    missing_classes = sum(value == 0 for value in water) + sum(
        value == 0 for value in non_water
    )
    sample_deviation = sum(
        abs(samples[index] - ratios[index] * total_samples)
        for index in range(3)
    ) / total_samples
    water_deviation = sum(
        abs(water[index] - ratios[index] * total_water)
        for index in range(3)
    ) / total_water
    non_water_deviation = sum(
        abs(non_water[index] - ratios[index] * total_non_water)
        for index in range(3)
    ) / total_non_water
    combined_deviation = sample_deviation + 0.5 * (
        water_deviation + non_water_deviation
    )
    # Arredondar evita que ruido de ponto flutuante decida empates conceituais.
    return (
        missing_classes,
        round(combined_deviation, 15),
        round(sample_deviation, 15),
        assignment,
    )


def choose_assignment(
    event_stats: list[EventStats],
    ratios: tuple[float, float, float],
) -> tuple[int, ...]:
    if len(event_stats) < len(SPLITS):
        raise InvalidDatasetError(
            "sao necessarios pelo menos tres eventos para criar tres splits"
        )
    if not any(stats.water for stats in event_stats):
        raise InvalidDatasetError("dataset nao possui exemplos WATER")
    if not any(stats.non_water for stats in event_stats):
        raise InvalidDatasetError("dataset nao possui exemplos NON_WATER")

    best_score: tuple[int, float, float, tuple[int, ...]] | None = None
    best_assignment: tuple[int, ...] | None = None
    for assignment in itertools.product(range(len(SPLITS)), repeat=len(event_stats)):
        if len(set(assignment)) != len(SPLITS):
            continue
        score = assignment_score(assignment, event_stats, ratios)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise InvalidDatasetError("nao foi possivel encontrar uma divisao valida")
    return best_assignment


def atomic_write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
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


def validate_splits(
    split_rows: dict[str, list[dict[str, object]]],
    total_rows: int,
) -> None:
    event_sets = {
        split: {str(row["event"]) for row in rows}
        for split, rows in split_rows.items()
    }
    id_sets = {
        split: {str(row["image_id"]).casefold() for row in rows}
        for split, rows in split_rows.items()
    }
    for left, right in itertools.combinations(SPLITS, 2):
        overlap_events = event_sets[left] & event_sets[right]
        if overlap_events:
            raise InvalidDatasetError(
                f"leakage de eventos entre {left} e {right}: "
                f"{sorted(overlap_events)}"
            )
        overlap_ids = id_sets[left] & id_sets[right]
        if overlap_ids:
            raise InvalidDatasetError(
                f"image_id repetido entre {left} e {right}: {sorted(overlap_ids)}"
            )
    combined_ids = set().union(*id_sets.values())
    combined_count = sum(len(rows) for rows in split_rows.values())
    if combined_count != total_rows or len(combined_ids) != total_rows:
        raise InvalidDatasetError(
            "a uniao dos splits nao coincide exatamente com o dataset: "
            f"splits={combined_count}, IDs unicos={len(combined_ids)}, "
            f"dataset={total_rows}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, input_fields = load_and_validate_dataset(args.dataset)
        event_stats = calculate_event_stats(rows)
        ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
        assignment = choose_assignment(event_stats, ratios)
    except (OSError, csv.Error, InvalidDatasetError) as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    event_to_split = {
        stats.event: SPLITS[assignment[index]]
        for index, stats in enumerate(event_stats)
    }
    split_rows: dict[str, list[dict[str, object]]] = {
        split: [] for split in SPLITS
    }
    for row in rows:
        output_row: dict[str, object] = dict(row)
        output_row["split"] = event_to_split[row["event"]]
        split_rows[str(output_row["split"])].append(output_row)
    for split in SPLITS:
        split_rows[split].sort(key=lambda row: str(row["image_id"]).casefold())

    try:
        validate_splits(split_rows, len(rows))
    except InvalidDatasetError as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    summary_rows: list[dict[str, object]] = []
    for split in SPLITS:
        current_rows = split_rows[split]
        events = sorted(
            {str(row["event"]) for row in current_rows}, key=str.casefold
        )
        water = sum(row["class"] == "WATER" for row in current_rows)
        non_water = sum(row["class"] == "NON_WATER" for row in current_rows)
        summary_rows.append(
            {
                "split": split,
                "events": ";".join(events),
                "samples": len(current_rows),
                "water": water,
                "non_water": non_water,
                "water_percentage": water / len(current_rows) * 100.0,
            }
        )

    output_dir = args.output_dir.resolve()
    output_fields = input_fields + (["split"] if "split" not in input_fields else [])
    try:
        for split in SPLITS:
            atomic_write_csv(
                output_dir / f"{split}.csv", output_fields, split_rows[split]
            )
        atomic_write_csv(
            output_dir / "split_summary.csv", SUMMARY_FIELDS, summary_rows
        )
    except OSError as exc:
        print(f"[ERRO FATAL] Nao foi possivel escrever os splits: {exc}", file=sys.stderr)
        return 1

    print("\nEstatisticas por evento (dataset binario)")
    for stats in event_stats:
        water_percentage = stats.water / stats.samples * 100.0
        print(
            f"{stats.event}: total={stats.samples}, WATER={stats.water}, "
            f"NON_WATER={stats.non_water}, WATER%={water_percentage:.2f}"
        )
    print(
        "\nDecisao: busca exaustiva deterministica; prioridade para ambas "
        "as classes e, depois, proximidade das proporcoes "
        f"{ratios[0] * 100:g}/{ratios[1] * 100:g}/{ratios[2] * 100:g}."
    )
    for summary in summary_rows:
        print(f"\n{str(summary['split']).upper()}")
        print(f"eventos: {str(summary['events']).replace(';', ', ')}")
        print(f"amostras: {summary['samples']}")
        print(f"WATER: {summary['water']}")
        print(f"NON_WATER: {summary['non_water']}")
    print("\nValidacoes: sem leakage de eventos; sem IDs repetidos; uniao completa.")
    print(f"Arquivos gerados em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
