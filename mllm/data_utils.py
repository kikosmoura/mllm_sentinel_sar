"""Leitura e validacao dos splits usados pela MLLM."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


REPRESENTATIONS = ("pseudo_rgb", "vv", "vh")
REPRESENTATION_COLUMNS = {
    "pseudo_rgb": "pseudo_rgb_image",
    "vv": "vv_image",
    "vh": "vh_image",
}
CLASS_TO_ID = {"NON_WATER": 0, "WATER": 1}
REQUIRED_COLUMNS = {
    "image_id",
    "event",
    "class",
    "class_id",
    "vv_image",
    "vh_image",
    "pseudo_rgb_image",
}


class MLLMDataError(ValueError):
    """Erro de integridade dos dados multimodais."""


@dataclass(frozen=True)
class MLLMExample:
    image_id: str
    event: str
    class_name: str
    class_id: int
    split: str
    representation: str
    image_path: Path


def load_split(
    path: Path,
    split_name: str,
    representation: str,
    validate_images: bool = True,
) -> list[MLLMExample]:
    if representation not in REPRESENTATIONS:
        raise MLLMDataError(f"representacao invalida: {representation!r}")
    if not path.is_file():
        raise MLLMDataError(f"CSV inexistente: {path}")

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise MLLMDataError(f"CSV sem cabecalho: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise MLLMDataError(
                f"{path}: coluna(s) ausente(s): {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise MLLMDataError(f"split {split_name} vazio: {path}")

    image_column = REPRESENTATION_COLUMNS[representation]
    examples: list[MLLMExample] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        image_id = row.get("image_id", "").strip()
        event = row.get("event", "").strip()
        class_name = row.get("class", "").strip()
        if not image_id or not event:
            raise MLLMDataError(
                f"{path}:{row_number}: image_id ou event vazio"
            )
        image_key = image_id.casefold()
        if image_key in seen_ids:
            raise MLLMDataError(f"image_id duplicado em {split_name}: {image_id}")
        seen_ids.add(image_key)
        if class_name not in CLASS_TO_ID:
            raise MLLMDataError(
                f"classe invalida para {image_id}: {class_name!r}"
            )
        try:
            class_id = int(row.get("class_id", ""))
        except ValueError as exc:
            raise MLLMDataError(
                f"class_id invalido para {image_id}: {row.get('class_id')!r}"
            ) from exc
        if class_id != CLASS_TO_ID[class_name]:
            raise MLLMDataError(
                f"classe/class_id inconsistentes para {image_id}: "
                f"{class_name}/{class_id}"
            )
        if row.get("split", split_name).strip() != split_name:
            raise MLLMDataError(
                f"split declarado incorretamente para {image_id}: "
                f"{row.get('split')!r}"
            )
        inferred_event, separator, chip_component = image_id.rpartition("_")
        if not separator or not chip_component or inferred_event != event:
            raise MLLMDataError(
                f"evento inconsistente para {image_id}: {event!r}"
            )

        image_path = Path(row.get(image_column, "")).expanduser()
        if not image_path.is_absolute():
            image_path = path.resolve().parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise MLLMDataError(
                f"imagem {representation} inexistente para {image_id}: {image_path}"
            )
        if validate_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                    if image.format != "PNG":
                        raise MLLMDataError(f"imagem nao e PNG: {image_path}")
            except MLLMDataError:
                raise
            except Exception as exc:
                raise MLLMDataError(
                    f"imagem invalida para {image_id}: {image_path}: {exc}"
                ) from exc
        examples.append(
            MLLMExample(
                image_id=image_id,
                event=event,
                class_name=class_name,
                class_id=class_id,
                split=split_name,
                representation=representation,
                image_path=image_path,
            )
        )
    return examples


def validate_geographic_isolation(
    splits: dict[str, Sequence[MLLMExample]],
) -> None:
    names = list(splits)
    id_sets = {
        name: {example.image_id.casefold() for example in examples}
        for name, examples in splits.items()
    }
    event_sets = {
        name: {example.event for example in examples}
        for name, examples in splits.items()
    }
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlapping_ids = id_sets[left] & id_sets[right]
            if overlapping_ids:
                raise MLLMDataError(
                    f"image_id em {left} e {right}: {sorted(overlapping_ids)}"
                )
            overlapping_events = event_sets[left] & event_sets[right]
            if overlapping_events:
                raise MLLMDataError(
                    f"evento em {left} e {right}: {sorted(overlapping_events)}"
                )
    for name, examples in splits.items():
        classes = {example.class_id for example in examples}
        if classes != {0, 1}:
            raise MLLMDataError(
                f"split {name} nao contem as duas classes: {sorted(classes)}"
            )


def atomic_write_csv(
    output_path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, object]],
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
        temporary_path.unlink(missing_ok=True)
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
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
