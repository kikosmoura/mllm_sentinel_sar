#!/usr/bin/env python3
"""Prepara JSONL de treino/validacao sem incluir o conjunto de teste."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from mllm.data_utils import (
    REPRESENTATIONS,
    MLLMDataError,
    MLLMExample,
    atomic_write_jsonl,
    load_split,
    validate_geographic_isolation,
)
from mllm.prompts import CLASSIFICATION_PROMPT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cria train.jsonl e validation.jsonl para LoRA/QLoRA. O teste e "
            "somente auditado quanto a leakage e nunca entra na saida."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.csv"),
        help="Split geografico usado no treinamento.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/splits/validation.csv"),
        help="Split usado somente para validacao durante o treinamento.",
    )
    parser.add_argument(
        "--test-audit",
        type=Path,
        default=Path("data/splits/test.csv"),
        help="Teste lido somente para verificar isolamento; nunca exportado.",
    )
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="pseudo_rgb",
        help="Representacao SAR gravada nos exemplos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mllm"),
        help="Diretorio de train.jsonl e validation.jsonl.",
    )
    args = parser.parse_args(argv)
    for argument_name in ("train", "validation", "test_audit"):
        path = getattr(args, argument_name)
        if not path.is_file():
            parser.error(
                f"--{argument_name.replace('_', '-')} nao e arquivo acessivel: {path}"
            )
    return args


def jsonl_record(example: MLLMExample) -> dict[str, object]:
    return {
        "image_id": example.image_id,
        "image": str(example.image_path),
        "representation": example.representation,
        "messages": [
            {"role": "user", "content": CLASSIFICATION_PROMPT},
            {"role": "assistant", "content": example.class_name},
        ],
    }


def validate_records(
    records: list[dict[str, object]],
    expected_examples: list[MLLMExample],
) -> None:
    if len(records) != len(expected_examples):
        raise MLLMDataError(
            f"registros={len(records)} difere de exemplos={len(expected_examples)}"
        )
    for record, example in zip(records, expected_examples, strict=True):
        if record["image_id"] != example.image_id:
            raise MLLMDataError("ordem/ID inconsistente no JSONL")
        if not Path(str(record["image"])).is_file():
            raise MLLMDataError(f"imagem inexistente no JSONL: {record['image']}")
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            raise MLLMDataError(f"messages invalido para {example.image_id}")
        if messages[0] != {"role": "user", "content": CLASSIFICATION_PROMPT}:
            raise MLLMDataError(f"prompt divergente para {example.image_id}")
        if messages[1] != {"role": "assistant", "content": example.class_name}:
            raise MLLMDataError(f"label divergente para {example.image_id}")
        serialized = json.dumps(messages[0], ensure_ascii=False)
        forbidden_values = (example.event, "water_percentage", "LabelHand", "mask")
        if any(value.casefold() in serialized.casefold() for value in forbidden_values):
            raise MLLMDataError(
                f"prompt contem metadado proibido para {example.image_id}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        splits = {
            "train": load_split(args.train, "train", args.representation),
            "validation": load_split(
                args.validation, "validation", args.representation
            ),
            "test": load_split(
                args.test_audit, "test", args.representation, validate_images=False
            ),
        }
        validate_geographic_isolation(splits)
        train_records = [jsonl_record(example) for example in splits["train"]]
        validation_records = [
            jsonl_record(example) for example in splits["validation"]
        ]
        validate_records(train_records, splits["train"])
        validate_records(validation_records, splits["validation"])
        output_dir = args.output_dir.resolve()
        atomic_write_jsonl(output_dir / "train.jsonl", train_records)
        atomic_write_jsonl(
            output_dir / "validation.jsonl", validation_records
        )
    except (OSError, ValueError, MLLMDataError) as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    print("\nDataset MLLM preparado")
    print(f"Treino: {len(train_records)} exemplos")
    print(f"Validation: {len(validation_records)} exemplos")
    print(f"Test auditado e nao exportado: {len(splits['test'])} exemplos")
    print(f"Representacao: {args.representation}")
    print("Prompt-base compartilhado: validado")
    print("Leakage de eventos/IDs: nenhum")
    print(f"Saida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
