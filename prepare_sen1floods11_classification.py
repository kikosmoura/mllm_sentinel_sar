#!/usr/bin/env python3
"""Prepara um CSV de classificacao a partir do Sen1Floods11 S1Hand/LabelHand.

O pareamento e feito pelo identificador do arquivo. Os sufixos conhecidos
``_S1Hand`` e ``_LabelHand`` sao removidos antes da comparacao.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioIOError
except ImportError as exc:  # pragma: no cover - depende do ambiente de execucao
    raise SystemExit(
        "Dependencia ausente. Instale 'numpy' e 'rasterio' antes de executar "
        f"este script ({exc})."
    ) from exc


SUPPORTED_EXTENSIONS = {".tif", ".tiff"}
CSV_FIELDS = [
    "image_id",
    "s1_path",
    "label_path",
    "water_pixels",
    "non_water_pixels",
    "valid_pixels",
    "nodata_pixels",
    "water_fraction",
    "water_percentage",
    "class",
]
CLASS_ORDER = ("NON_WATER", "AMBIGUOUS", "WATER")


class InvalidPairError(ValueError):
    """Erro de validacao restrito a um par S1/mascara."""


@dataclass(frozen=True)
class RasterFile:
    image_id: str
    path: Path


@dataclass(frozen=True)
class ProcessingError:
    image_id: str
    s1_path: Path
    label_path: Path
    message: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Associa chips S1Hand as mascaras LabelHand do Sen1Floods11 e "
            "gera classes pela porcentagem de pixels de agua."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--s1-dir",
        type=Path,
        required=True,
        help="Diretorio S1Hand (a busca por GeoTIFFs e recursiva).",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        required=True,
        help="Diretorio LabelHand (a busca por GeoTIFFs e recursiva).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sen1floods11_classification.csv"),
        help="Caminho do CSV de saida.",
    )
    parser.add_argument(
        "--non-water-threshold",
        type=float,
        default=1.0,
        help=(
            "Limite percentual inclusivo para NON_WATER "
            "(water_percentage <= limite)."
        ),
    )
    parser.add_argument(
        "--water-threshold",
        type=float,
        default=10.0,
        help=(
            "Limite percentual inclusivo para WATER "
            "(water_percentage >= limite)."
        ),
    )

    args = parser.parse_args(argv)
    for name, value in (
        ("--non-water-threshold", args.non_water_threshold),
        ("--water-threshold", args.water_threshold),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 100.0:
            parser.error(f"{name} deve ser um numero finito entre 0 e 100")
    if args.non_water_threshold >= args.water_threshold:
        parser.error(
            "--non-water-threshold deve ser menor que --water-threshold"
        )

    for name, directory in (
        ("--s1-dir", args.s1_dir),
        ("--label-dir", args.label_dir),
    ):
        if not directory.is_dir():
            parser.error(f"{name} nao e um diretorio acessivel: {directory}")

    return args


def discover_geotiffs(directory: Path) -> list[Path]:
    """Retorna GeoTIFFs em ordem deterministica, incluindo subdiretorios."""
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def extract_image_id(path: Path, role_suffix: str) -> str:
    """Extrai o ID, aceitando nomes com ou sem o sufixo de papel conhecido."""
    image_id = path.stem
    if image_id.casefold().endswith(role_suffix.casefold()):
        image_id = image_id[: -len(role_suffix)]
    return image_id.strip()


def build_index(
    paths: Iterable[Path], role_suffix: str
) -> tuple[dict[str, list[RasterFile]], list[Path]]:
    """Indexa por ID sem diferenciar maiusculas; preserva duplicidades."""
    index: dict[str, list[RasterFile]] = defaultdict(list)
    invalid_names: list[Path] = []
    for path in paths:
        image_id = extract_image_id(path, role_suffix)
        if not image_id:
            invalid_names.append(path)
            continue
        index[image_id.casefold()].append(RasterFile(image_id, path.resolve()))
    return dict(index), invalid_names


def abbreviated_values(values: np.ndarray, limit: int = 10) -> str:
    """Formata valores invalidos sem produzir mensagens enormes."""
    unique_values = np.unique(values)
    shown = [repr(value.item()) for value in unique_values[:limit]]
    if unique_values.size > limit:
        shown.append(f"... (+{unique_values.size - limit} valores distintos)")
    return ", ".join(shown)


def inspect_pair(
    image_id: str,
    s1_path: Path,
    label_path: Path,
    non_water_threshold: float,
    water_threshold: float,
) -> dict[str, object]:
    """Valida um par, conta os valores da mascara e monta uma linha do CSV."""
    try:
        with rasterio.open(s1_path) as s1_dataset:
            if s1_dataset.count < 2:
                raise InvalidPairError(
                    "imagem S1 deve ter pelo menos duas bandas (VV e VH), "
                    f"mas possui {s1_dataset.count}"
                )
            s1_shape = (s1_dataset.height, s1_dataset.width)
    except RasterioIOError as exc:
        raise InvalidPairError(f"nao foi possivel abrir a imagem S1: {exc}") from exc
    except OSError as exc:
        raise InvalidPairError(f"erro de leitura na imagem S1: {exc}") from exc

    try:
        with rasterio.open(label_path) as label_dataset:
            if label_dataset.count != 1:
                raise InvalidPairError(
                    "mascara deve ter exatamente uma banda, "
                    f"mas possui {label_dataset.count}"
                )
            label_shape = (label_dataset.height, label_dataset.width)
            if label_shape != s1_shape:
                raise InvalidPairError(
                    "dimensoes incompativeis: "
                    f"S1={s1_shape[1]}x{s1_shape[0]}, "
                    f"mascara={label_shape[1]}x{label_shape[0]}"
                )
            mask = label_dataset.read(1, masked=False)
    except RasterioIOError as exc:
        raise InvalidPairError(f"nao foi possivel abrir a mascara: {exc}") from exc
    except OSError as exc:
        raise InvalidPairError(f"erro de leitura na mascara: {exc}") from exc

    expected_values = np.isin(mask, (-1, 0, 1))
    if not bool(np.all(expected_values)):
        unexpected = mask[~expected_values]
        raise InvalidPairError(
            f"mascara contem {unexpected.size} pixel(is) com valor(es) "
            f"inesperado(s): {abbreviated_values(unexpected)}"
        )

    water_pixels = int(np.count_nonzero(mask == 1))
    non_water_pixels = int(np.count_nonzero(mask == 0))
    nodata_pixels = int(np.count_nonzero(mask == -1))
    valid_pixels = water_pixels + non_water_pixels

    if valid_pixels == 0:
        raise InvalidPairError("mascara nao possui pixels validos (0 ou 1)")
    if water_pixels + non_water_pixels != valid_pixels:
        raise InvalidPairError("contagem interna inconsistente de pixels validos")
    if valid_pixels + nodata_pixels != int(mask.size):
        raise InvalidPairError("contagem interna inconsistente do total de pixels")

    water_fraction = water_pixels / valid_pixels
    if not 0.0 <= water_fraction <= 1.0:
        raise InvalidPairError(
            f"fracao de agua fora do intervalo [0, 1]: {water_fraction}"
        )
    water_percentage = water_fraction * 100.0

    if water_percentage <= non_water_threshold:
        chip_class = "NON_WATER"
    elif water_percentage >= water_threshold:
        chip_class = "WATER"
    else:
        chip_class = "AMBIGUOUS"

    return {
        "image_id": image_id,
        "s1_path": str(s1_path),
        "label_path": str(label_path),
        "water_pixels": water_pixels,
        "non_water_pixels": non_water_pixels,
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_pixels,
        "water_fraction": water_fraction,
        "water_percentage": water_percentage,
        "class": chip_class,
    }


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except Exception:
        # Evita deixar um CSV parcial caso a escrita falhe.
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def print_paths(title: str, paths: list[Path]) -> None:
    print(f"{title}: {len(paths)}")
    for path in paths:
        print(f"  - {path}")


def print_report(
    s1_files: list[Path],
    label_files: list[Path],
    pair_count: int,
    rows: list[dict[str, object]],
    missing_labels: list[Path],
    missing_images: list[Path],
    duplicate_s1: dict[str, list[RasterFile]],
    duplicate_labels: dict[str, list[RasterFile]],
    invalid_names: list[Path],
    errors: list[ProcessingError],
    output_path: Path,
) -> None:
    print("\nRelatorio Sen1Floods11")
    print(f"Arquivos S1 encontrados: {len(s1_files)}")
    print(f"Mascaras encontradas: {len(label_files)}")
    print(f"Total de pares encontrados: {pair_count}")
    print(f"Total processado: {len(rows)}")
    print_paths("Arquivos sem mascara correspondente", missing_labels)
    print_paths("Mascaras sem imagem Sentinel-1 correspondente", missing_images)
    print(
        "Identificadores duplicados: "
        f"{len(duplicate_s1) + len(duplicate_labels)} "
        f"(S1={len(duplicate_s1)}, mascaras={len(duplicate_labels)})"
    )
    print(f"Arquivos com identificador vazio: {len(invalid_names)}")
    print(f"Erros em pares encontrados: {len(errors)}")

    class_counts = Counter(str(row["class"]) for row in rows)
    print("Distribuicao das classes:")
    for chip_class in CLASS_ORDER:
        count = class_counts[chip_class]
        percentage = (count / len(rows) * 100.0) if rows else 0.0
        print(f"  {chip_class}: {count} ({percentage:.2f}%)")

    percentages = [float(row["water_percentage"]) for row in rows]
    print("Estatisticas de water_percentage:")
    if percentages:
        print(f"  media: {statistics.fmean(percentages):.6f}%")
        print(f"  mediana: {statistics.median(percentages):.6f}%")
        print(f"  minimo: {min(percentages):.6f}%")
        print(f"  maximo: {max(percentages):.6f}%")
    else:
        print("  media: N/A")
        print("  mediana: N/A")
        print("  minimo: N/A")
        print("  maximo: N/A")
    print(f"CSV gerado: {output_path.resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    s1_files = discover_geotiffs(args.s1_dir)
    label_files = discover_geotiffs(args.label_dir)
    s1_index, invalid_s1_names = build_index(s1_files, "_S1Hand")
    label_index, invalid_label_names = build_index(label_files, "_LabelHand")
    invalid_names = invalid_s1_names + invalid_label_names

    duplicate_s1 = {
        key: files for key, files in s1_index.items() if len(files) > 1
    }
    duplicate_labels = {
        key: files for key, files in label_index.items() if len(files) > 1
    }

    for path in invalid_names:
        print(f"[ERRO] Identificador vazio: {path.resolve()}", file=sys.stderr)
    for role, duplicates in (("S1", duplicate_s1), ("mascara", duplicate_labels)):
        for files in duplicates.values():
            paths = ", ".join(str(item.path) for item in files)
            print(
                f"[ERRO] Identificador duplicado em {role} "
                f"({files[0].image_id!r}): {paths}",
                file=sys.stderr,
            )

    s1_keys = set(s1_index)
    label_keys = set(label_index)
    missing_labels = sorted(
        (item.path for key in s1_keys - label_keys for item in s1_index[key]),
        key=lambda path: path.as_posix().casefold(),
    )
    missing_images = sorted(
        (item.path for key in label_keys - s1_keys for item in label_index[key]),
        key=lambda path: path.as_posix().casefold(),
    )
    for path in missing_labels:
        print(f"[ERRO] Imagem S1 sem mascara: {path}", file=sys.stderr)
    for path in missing_images:
        print(f"[ERRO] Mascara sem imagem S1: {path}", file=sys.stderr)

    pair_keys = sorted(
        (
            key
            for key in s1_keys & label_keys
            if len(s1_index[key]) == 1 and len(label_index[key]) == 1
        )
    )

    rows: list[dict[str, object]] = []
    errors: list[ProcessingError] = []
    for key in pair_keys:
        s1_file = s1_index[key][0]
        label_file = label_index[key][0]
        try:
            row = inspect_pair(
                image_id=s1_file.image_id,
                s1_path=s1_file.path,
                label_path=label_file.path,
                non_water_threshold=args.non_water_threshold,
                water_threshold=args.water_threshold,
            )
        except Exception as exc:
            # A fronteira por par e intencional: um GeoTIFF defeituoso ou uma
            # falha inesperada em um chip nao deve descartar os demais chips.
            error = ProcessingError(
                image_id=s1_file.image_id,
                s1_path=s1_file.path,
                label_path=label_file.path,
                message=str(exc),
            )
            errors.append(error)
            print(
                f"[ERRO] Par {error.image_id!r}: {error.message}; "
                f"S1={error.s1_path}; mascara={error.label_path}",
                file=sys.stderr,
            )
            continue
        rows.append(row)

    try:
        write_csv(args.output, rows)
    except OSError as exc:
        print(f"[ERRO FATAL] Nao foi possivel escrever o CSV: {exc}", file=sys.stderr)
        return 1

    print_report(
        s1_files=s1_files,
        label_files=label_files,
        pair_count=len(pair_keys),
        rows=rows,
        missing_labels=missing_labels,
        missing_images=missing_images,
        duplicate_s1=duplicate_s1,
        duplicate_labels=duplicate_labels,
        invalid_names=invalid_names,
        errors=errors,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
