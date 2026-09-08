#!/usr/bin/env python3
"""Cria o dataset binario WATER x NON_WATER sem alterar o CSV original."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depende do ambiente
    raise SystemExit(
        "Dependencia ausente. Instale 'Pillow' antes de executar este script "
        f"({exc})."
    ) from exc


REQUIRED_COLUMNS = {
    "image_id",
    "class",
    "water_percentage",
    "s1_path",
}
BINARY_CLASS_IDS = {"NON_WATER": 0, "WATER": 1}
KNOWN_CLASSES = set(BINARY_CLASS_IDS) | {"AMBIGUOUS"}
OUTPUT_FIELDS = [
    "image_id",
    "event",
    "class",
    "class_id",
    "water_percentage",
    "s1_path",
    "vv_image",
    "vh_image",
    "pseudo_rgb_image",
]


class InvalidDatasetError(ValueError):
    """Erro de consistencia do dataset experimental."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cria o dataset experimental binario, mapeando NON_WATER=0 e "
            "WATER=1 e excluindo AMBIGUOUS somente da saida principal."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("sen1floods11_classification.csv"),
        help="CSV original com as classes derivadas das mascaras.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Diretorio que contem vv/, vh/ e pseudo_rgb/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experimental_dataset.csv"),
        help="CSV binario de saida.",
    )
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        parser.error(f"--csv nao e um arquivo acessivel: {args.csv}")
    if not args.images_dir.is_dir():
        parser.error(f"--images-dir nao e um diretorio acessivel: {args.images_dir}")
    return args


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InvalidDatasetError("CSV sem cabecalho")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise InvalidDatasetError(
                "CSV sem coluna(s) obrigatoria(s): " + ", ".join(sorted(missing))
            )
        return list(reader)


def safe_image_id(raw_image_id: str) -> str:
    image_id = raw_image_id.strip()
    if not image_id or image_id in {".", ".."}:
        raise InvalidDatasetError("image_id vazio ou invalido")
    if Path(image_id).name != image_id or "/" in image_id or "\\" in image_id:
        raise InvalidDatasetError(
            f"image_id contem separador de caminho: {image_id!r}"
        )
    return image_id


def extract_event(image_id: str) -> str:
    """Remove apenas o ultimo componente numerico; preserva hifens no evento."""
    event, separator, chip_id = image_id.rpartition("_")
    if not separator or not event or not chip_id:
        raise InvalidDatasetError(
            f"nao foi possivel extrair o evento de image_id={image_id!r}"
        )
    return event


def resolve_source_path(raw_path: str, csv_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = csv_path.resolve().parent / path
    return path.resolve()


def validate_png_triplet(paths: dict[str, Path]) -> tuple[int, int]:
    dimensions: set[tuple[int, int]] = set()
    for representation, path in paths.items():
        if not path.is_file():
            raise InvalidDatasetError(
                f"PNG {representation} inexistente: {path}"
            )
        try:
            with Image.open(path) as image:
                image.load()
                if image.format != "PNG":
                    raise InvalidDatasetError(
                        f"arquivo {path} nao esta no formato PNG"
                    )
                if image.mode != "RGB":
                    raise InvalidDatasetError(
                        f"PNG {path} tem modo {image.mode}; esperado RGB"
                    )
                dimensions.add(image.size)
                extrema = image.getextrema()
                if any(low < 0 or high > 255 for low, high in extrema):
                    raise InvalidDatasetError(
                        f"PNG {path} contem valor fora de [0, 255]"
                    )
        except InvalidDatasetError:
            raise
        except Exception as exc:
            raise InvalidDatasetError(f"PNG invalido {path}: {exc}") from exc
    if len(dimensions) != 1:
        raise InvalidDatasetError(
            f"representacoes do chip possuem dimensoes distintas: {dimensions}"
        )
    return next(iter(dimensions))


def atomic_write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_rows = load_rows(args.csv)
    except (OSError, csv.Error, InvalidDatasetError) as exc:
        print(f"[ERRO FATAL] Nao foi possivel ler o CSV: {exc}", file=sys.stderr)
        return 1

    source_counts = Counter(row.get("class", "").strip().upper() for row in source_rows)
    expected_binary_count = sum(source_counts[name] for name in BINARY_CLASS_IDS)
    ambiguous_count = source_counts["AMBIGUOUS"]

    id_counts = Counter(row.get("image_id", "").strip().casefold() for row in source_rows)
    duplicate_ids = {key for key, count in id_counts.items() if key and count > 1}
    images_root = args.images_dir.resolve()
    output_rows: list[dict[str, object]] = []
    errors: list[str] = []
    expected_dimensions: tuple[int, int] | None = None

    for row in source_rows:
        raw_image_id = row.get("image_id", "")
        raw_class = row.get("class", "").strip().upper()
        try:
            if raw_class not in KNOWN_CLASSES:
                raise InvalidDatasetError(
                    f"classe inesperada {raw_class!r} em {raw_image_id!r}"
                )
            if raw_class == "AMBIGUOUS":
                continue

            image_id = safe_image_id(raw_image_id)
            if image_id.casefold() in duplicate_ids:
                raise InvalidDatasetError(f"image_id duplicado: {image_id!r}")
            event = extract_event(image_id)
            water_percentage = float(row.get("water_percentage", ""))
            if not math.isfinite(water_percentage) or not 0.0 <= water_percentage <= 100.0:
                raise InvalidDatasetError(
                    f"water_percentage invalido em {image_id!r}: "
                    f"{row.get('water_percentage')!r}"
                )

            s1_path = resolve_source_path(row.get("s1_path", ""), args.csv)
            if not s1_path.is_file():
                raise InvalidDatasetError(f"arquivo S1 inexistente: {s1_path}")
            image_paths = {
                "vv": (images_root / "vv" / f"{image_id}.png").resolve(),
                "vh": (images_root / "vh" / f"{image_id}.png").resolve(),
                "pseudo_rgb": (
                    images_root / "pseudo_rgb" / f"{image_id}.png"
                ).resolve(),
            }
            dimensions = validate_png_triplet(image_paths)
            if expected_dimensions is None:
                expected_dimensions = dimensions
            elif dimensions != expected_dimensions:
                raise InvalidDatasetError(
                    f"dimensoes {dimensions} de {image_id!r} diferem do "
                    f"padrao {expected_dimensions}"
                )

            output_rows.append(
                {
                    "image_id": image_id,
                    "event": event,
                    "class": raw_class,
                    "class_id": BINARY_CLASS_IDS[raw_class],
                    "water_percentage": row["water_percentage"],
                    "s1_path": str(s1_path),
                    "vv_image": str(image_paths["vv"]),
                    "vh_image": str(image_paths["vh"]),
                    "pseudo_rgb_image": str(image_paths["pseudo_rgb"]),
                }
            )
        except Exception as exc:
            message = f"Registro {raw_image_id!r}: {exc}"
            errors.append(message)

    output_rows.sort(key=lambda row: str(row["image_id"]).casefold())
    output_classes = {str(row["class"]) for row in output_rows}
    output_class_ids = {int(row["class_id"]) for row in output_rows}
    output_ids = [str(row["image_id"]).casefold() for row in output_rows]

    if output_classes - set(BINARY_CLASS_IDS):
        errors.append(f"classes invalidas na saida: {sorted(output_classes)}")
    if output_class_ids - {0, 1}:
        errors.append(f"class_id invalidos na saida: {sorted(output_class_ids)}")
    if len(output_ids) != len(set(output_ids)):
        errors.append("image_id duplicado na saida")
    if len(output_rows) != expected_binary_count:
        errors.append(
            "quantidade binaria inconsistente: "
            f"saida={len(output_rows)}, esperado={expected_binary_count}"
        )
    for class_name in source_counts:
        if class_name not in KNOWN_CLASSES:
            errors.append(f"classe inesperada no CSV original: {class_name!r}")

    print("\nRelatorio do dataset experimental")
    print(f"WATER: {source_counts['WATER']}")
    print(f"NON_WATER: {source_counts['NON_WATER']}")
    print(f"AMBIGUOUS excluidos do experimento principal: {ambiguous_count}")
    print(f"Registros binarios esperados: {expected_binary_count}")
    print(f"Registros binarios preparados: {len(output_rows)}")
    print(f"Dimensoes dos PNGs: {expected_dimensions or 'N/A'}")
    print(f"Erros encontrados: {len(errors)}")

    if errors:
        for message in errors:
            print(f"[ERRO] {message}", file=sys.stderr)
        print("CSV de saida nao foi substituido devido aos erros.", file=sys.stderr)
        return 1

    try:
        atomic_write_csv(args.output, output_rows)
    except OSError as exc:
        print(f"[ERRO FATAL] Nao foi possivel escrever a saida: {exc}", file=sys.stderr)
        return 1
    print(f"CSV gerado: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
