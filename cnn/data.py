"""Pixels como entradas; auditoria geografica sem interpretar rotulos de teste."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from evaluation.metrics import GroundTruth, load_ground_truth


REPRESENTATIONS = ("pseudo_rgb", "vv", "vh")
DEFAULT_PREPROCESSING = {
    "size": [224, 224],
    "resize": "full-chip",
    "interpolation": "bilinear",
    "antialias": True,
    "input_mode": "RGB",
    "scale": "uint8_to_0_1",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "training_augmentation": {"horizontal_flip": 0.5, "vertical_flip": 0.5},
}


class CNNError(ValueError):
    """Dados ou artefatos inconsistentes: nao produzir resultados parciais."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_splits(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Le apenas ID/event/split; bytes dos CSVs sao usados para hashes opacos.

    Nenhum rotulo, porcentagem ou caminho de imagem de teste e interpretado.
    O treinamento pode assim comprovar isolamento sem carregar o teste.
    """
    if set(paths) != {"train", "validation", "test"}:
        raise CNNError("auditoria exige train, validation e test")
    result = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        reader = csv.reader(io.StringIO(raw.decode("utf-8")))
        header = next(reader, [])
        if len(header) != len(set(header)):
            raise CNNError(f"cabecalho duplicado: {path}")
        if not {"image_id", "event", "split"}.issubset(header):
            raise CNNError(f"ID/event/split ausente: {path}")
        columns = {key: header.index(key) for key in ("image_id", "event", "split")}
        ids, events = [], []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise CNNError(f"linha CSV malformada: {path}:{line}")
            image_id, event, split = (row[columns[key]].strip() for key in columns)
            inferred, sep, chip = image_id.rpartition("_")
            if not event or not sep or not chip or inferred != event or split != name:
                raise CNNError(f"ID/event/split inconsistente: {path}:{line}")
            ids.append(image_id)
            events.append(event)
        if not ids or len({value.casefold() for value in ids}) != len(ids):
            raise CNNError(f"split vazio ou IDs duplicados: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "samples": len(ids),
            "image_ids": ids,
            "events": sorted(set(events)),
        }
    validate_split_audit(result)
    return result


def validate_split_audit(result: Mapping) -> None:
    """Valida uma auditoria persistida sem abrir os CSVs que ela descreve."""
    if set(result) != {"train", "validation", "test"}:
        raise CNNError("auditoria exige train, validation e test")
    for name, entry in result.items():
        ids, events = entry["image_ids"], entry["events"]
        digest = entry["sha256"]
        if (not ids or entry["samples"] != len(ids)
                or len({item.casefold() for item in ids}) != len(ids)
                or set(events) != {item.rpartition("_")[0] for item in ids}
                or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
            raise CNNError(f"auditoria persistida invalida: {name}")
    names = list(result)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            for key in ("image_ids", "events"):
                overlap = {s.casefold() for s in result[left][key]} & {
                    s.casefold() for s in result[right][key]
                }
                if overlap:
                    raise CNNError(f"{key} compartilhados em {left}/{right}: {sorted(overlap)}")


def verify_split_hashes(paths: Mapping[str, Path], audit: Mapping) -> None:
    for name, path in paths.items():
        if file_sha256(path) != audit[name]["sha256"]:
            raise CNNError(f"CSV alterado desde a auditoria: {path}")


def verify_image_hashes(examples: Sequence[SARExample]) -> None:
    for item in examples:
        if file_sha256(item.image_path) != item.image_sha256:
            raise CNNError(f"PNG alterado durante a execucao: {item.image_path}")


@dataclass(frozen=True)
class SARExample:
    truth: GroundTruth
    image_path: Path
    image_sha256: str


def resolve_png(raw_path: str, image_id: str, representation: str,
                project_root: Path, csv_path: Path) -> tuple[Path, dict[str, object]]:
    """Prefere a copia local, exige identidade SHA256 de todas as copias existentes."""
    if not raw_path.strip():
        raise CNNError(f"caminho PNG vazio: {image_id}")
    declared = Path(raw_path).expanduser()
    expected_name = f"{image_id}.png"
    if declared.name != expected_name or declared.parent.name != representation:
        raise CNNError(f"PNG nao corresponde ao ID/representacao: {declared}")
    local = project_root / "data" / "images" / representation / expected_name
    candidates = [local]
    if declared.is_absolute():
        candidates.append(declared)
    else:
        candidates.extend((project_root / declared, csv_path.resolve().parent / declared))
    existing = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if not existing:
        raise CNNError(f"PNG inexistente para {image_id}: {candidates}")
    hashes = {str(path): file_sha256(path) for path in existing}
    if len(set(hashes.values())) != 1:
        raise CNNError(f"copias divergentes do PNG para {image_id}: {hashes}")
    selected = existing[0]
    return selected, {
        "image_id": image_id,
        "declared_path": raw_path,
        "resolved_path": str(selected),
        "sha256": hashes[str(selected)],
        "verified_copies": hashes,
    }


def load_examples(path: Path, split: str, representation: str,
                  project_root: Path) -> tuple[list[SARExample], list[dict[str, object]]]:
    if representation not in REPRESENTATIONS:
        raise CNNError(f"representacao desconhecida: {representation}")
    truth = load_ground_truth(path, expected_split=split)
    column = f"{representation}_image"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if column not in (reader.fieldnames or []):
            raise CNNError(f"coluna {column} ausente: {path}")
        rows = list(reader)
    examples, resolutions = [], []
    for item, row in zip(truth, rows, strict=True):
        selected, record = resolve_png(row[column], item.image_id, representation,
                                       project_root, path)
        # Verificacao completa, inclusive decodificacao; nunca excluir uma imagem.
        try:
            with Image.open(selected) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise CNNError(f"esperado PNG RGB: {selected}")
                image.load()
        except (OSError, SyntaxError) as exc:
            raise CNNError(f"erro de leitura PNG: {selected}: {exc}") from exc
        examples.append(SARExample(item, selected, str(record["sha256"])))
        resolutions.append(record)
    return examples, resolutions


def make_transform(preprocessing: Mapping, training: bool = False):
    # O esquema salvo e validado para impedir um crop/normalizacao inadvertido.
    if dict(preprocessing) != DEFAULT_PREPROCESSING:
        raise CNNError("pre-processamento salvo nao corresponde ao esquema suportado")
    steps = [transforms.Resize(tuple(preprocessing["size"]),
                               interpolation=InterpolationMode.BILINEAR,
                               antialias=preprocessing["antialias"])]
    if training:
        aug = preprocessing["training_augmentation"]
        steps.extend((transforms.RandomHorizontalFlip(aug["horizontal_flip"]),
                      transforms.RandomVerticalFlip(aug["vertical_flip"])))
    steps.extend((transforms.ToTensor(),
                  transforms.Normalize(preprocessing["mean"], preprocessing["std"])))
    return transforms.Compose(steps)


class SARDataset(Dataset):
    """Retorna somente tensor de pixels, rotulo e indice para rastreabilidade."""

    def __init__(self, examples: Sequence[SARExample], preprocessing: Mapping,
                 training: bool = False):
        self.examples = list(examples)
        self.transform = make_transform(preprocessing, training)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        try:
            with Image.open(item.image_path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise CNNError(f"esperado PNG RGB: {item.image_path}")
                tensor = self.transform(image)
        except (OSError, SyntaxError) as exc:
            raise CNNError(f"erro de leitura PNG: {item.image_path}: {exc}") from exc
        if not bool(torch.isfinite(tensor).all()):
            raise CNNError(f"pixels nao finitos: {item.image_path}")
        return tensor, item.truth.true_class_id, index
