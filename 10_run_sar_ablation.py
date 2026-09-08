#!/usr/bin/env python3
"""Orquestra a ablation VV/VH/pseudo-RGB sem repetir artefatos validos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from evaluation.metrics import EvaluationError, load_ground_truth, load_predictions
from mllm.model_utils import DEFAULT_MODEL_ID, parse_predicted_class
from mllm.prompts import CLASSIFICATION_PROMPT


REPRESENTATIONS = ("vv", "vh", "pseudo_rgb")
FIXED_TRAINING_KEYS = (
    "base_model",
    "requested_revision",
    "training_samples",
    "validation_samples",
    "test_samples_used",
    "epochs",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "seed",
    "quantization",
    "gradient_checkpointing",
    "do_image_splitting",
    "early_stopping_patience",
    "prompt_sha256",
)


class AblationError(RuntimeError):
    """Falha que invalida ou impede a ablation study controlada."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa zero-shot e um LoRA independente para VV, VH e pseudo_rgb, "
            "reutilizando somente artefatos existentes que passem nas validacoes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument(
        "--validation", type=Path, default=Path("data/splits/validation.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/ablation")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/mllm_ablation")
    )
    parser.add_argument(
        "--existing-pseudo-adapter",
        type=Path,
        default=Path("models/mllm_adapter_pseudo_rgb"),
    )
    parser.add_argument(
        "--existing-pseudo-zero-shot",
        type=Path,
        default=Path("results/mllm_zero_shot"),
    )
    parser.add_argument(
        "--existing-pseudo-finetuned",
        type=Path,
        default=Path("results/mllm_finetuned"),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--do-image-splitting",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    for name in ("train", "validation", "test"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name}: arquivo inexistente: {path}")
    if len(args.representations) != len(set(args.representations)):
        parser.error("--representations contem valores duplicados")
    positive = {
        "--epochs": args.epochs,
        "--learning-rate": args.learning_rate,
        "--batch-size": args.batch_size,
        "--gradient-accumulation-steps": args.gradient_accumulation_steps,
        "--lora-r": args.lora_r,
        "--lora-alpha": args.lora_alpha,
        "--max-new-tokens": args.max_new_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} deve ser maior que zero")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout deve estar em [0,1)")
    return args


def _read_json(path: Path) -> dict[str, object]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationError(f"JSON invalido {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise AblationError(f"objeto JSON esperado em {path}")
    return result


def _atomic_json(path: Path, payload: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], root: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise AblationError(
            f"comando falhou com codigo {exc.returncode}: {' '.join(command)}"
        ) from exc


def _metrics_metadata_valid(
    path: Path, model_id: str, representation: str
) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        return (
            {row.get("split") for row in rows} == {"validation", "test"}
            and all(row.get("model") == model_id for row in rows)
            and all(row.get("representation") == representation for row in rows)
        )
    except (OSError, csv.Error):
        return False


def _evaluation_valid(
    directory: Path,
    validation_path: Path,
    test_path: Path,
    model_id: str,
    representation: str,
) -> bool:
    try:
        if not _metrics_metadata_valid(
            directory / "metrics.csv", model_id, representation
        ):
            return False
        for split, split_path in (
            ("validation", validation_path),
            ("test", test_path),
        ):
            truth = load_ground_truth(split_path, expected_split=split)
            predictions = load_predictions(
                directory / f"{split}_predictions.csv",
                truth,
                model="MLLM",
                training="audit",
                representation=representation,
            )
            for row in predictions:
                parsed = parse_predicted_class(row.raw_response)
                if parsed != row.predicted_class:
                    return False
        return True
    except (EvaluationError, OSError, ValueError):
        return False


def _adapter_valid(
    adapter_dir: Path,
    args: argparse.Namespace,
    representation: str,
    train_samples: int,
    validation_samples: int,
) -> bool:
    config_path = adapter_dir / "training_config.json"
    if not (
        config_path.is_file()
        and (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "adapter_model.safetensors").is_file()
    ):
        return False
    try:
        config = _read_json(config_path)
    except AblationError:
        return False
    expected = {
        "base_model": args.model_id,
        "requested_revision": args.revision,
        "representation": representation,
        "training_samples": train_samples,
        "validation_samples": validation_samples,
        "test_samples_used": 0,
        "epochs": float(args.epochs),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "seed": args.seed,
        "quantization": "4-bit NF4" if args.use_4bit else "none",
        "gradient_checkpointing": args.gradient_checkpointing,
        "do_image_splitting": args.do_image_splitting,
        "early_stopping_patience": args.early_stopping_patience,
        "prompt_sha256": hashlib.sha256(
            CLASSIFICATION_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    return all(config.get(key) == value for key, value in expected.items())


def _common_evaluation_command(
    script: Path,
    args: argparse.Namespace,
    representation: str,
    output_dir: Path,
    environment_output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--model-id",
        args.model_id,
        "--train-audit",
        str(args.train),
        "--validation",
        str(args.validation),
        "--test",
        str(args.test),
        "--representation",
        representation,
        "--output-dir",
        str(output_dir),
        "--environment-output",
        str(environment_output),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--no-do-image-splitting"
        if not args.do_image_splitting
        else "--do-image-splitting",
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    if args.use_4bit:
        command.append("--use-4bit")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def _copy_prediction(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _check_only_representation_changes(configs: dict[str, dict[str, object]]) -> None:
    if len(configs) < 2:
        return
    names = list(configs)
    reference_name = names[0]
    reference = configs[reference_name]
    for name in names[1:]:
        candidate = configs[name]
        differences = [
            key for key in FIXED_TRAINING_KEYS if candidate.get(key) != reference.get(key)
        ]
        if differences:
            raise AblationError(
                f"configuracoes LoRA diferem entre {reference_name} e {name}: "
                f"{differences}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    train_truth = load_ground_truth(args.train, expected_split="train")
    validation_truth = load_ground_truth(
        args.validation, expected_split="validation"
    )
    test_truth = load_ground_truth(args.test, expected_split="test")
    all_ids = [
        {row.image_id for row in split}
        for split in (train_truth, validation_truth, test_truth)
    ]
    all_events = [
        {row.event for row in split}
        for split in (train_truth, validation_truth, test_truth)
    ]
    if any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise SystemExit("[ERRO FATAL] IDs compartilhados entre splits")
    if any(
        all_events[i] & all_events[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise SystemExit("[ERRO FATAL] eventos compartilhados entre splits")

    results_dir = args.results_dir.resolve()
    models_dir = args.models_dir.resolve()
    actions: dict[str, dict[str, str]] = {}
    adapter_configs: dict[str, dict[str, object]] = {}
    try:
        for representation in args.representations:
            actions[representation] = {}
            environment_output = results_dir / f"environment_{representation}.json"
            if representation == "pseudo_rgb":
                zero_dir = args.existing_pseudo_zero_shot.resolve()
                adapter_dir = args.existing_pseudo_adapter.resolve()
                finetuned_dir = args.existing_pseudo_finetuned.resolve()
            else:
                zero_dir = results_dir / f"zero_shot_{representation}"
                adapter_dir = models_dir / f"mllm_adapter_{representation}"
                finetuned_dir = results_dir / f"finetuned_{representation}"

            if _evaluation_valid(
                zero_dir,
                args.validation,
                args.test,
                args.model_id,
                representation,
            ):
                actions[representation]["zero_shot"] = "reused"
            else:
                zero_command = _common_evaluation_command(
                    root / "06_evaluate_mllm_zero_shot.py",
                    args,
                    representation,
                    zero_dir,
                    environment_output,
                )
                _run(zero_command, root)
                if not _evaluation_valid(
                    zero_dir,
                    args.validation,
                    args.test,
                    args.model_id,
                    representation,
                ):
                    raise AblationError(
                        f"saida zero-shot invalida para {representation}"
                    )
                actions[representation]["zero_shot"] = "executed"

            if not _adapter_valid(
                adapter_dir,
                args,
                representation,
                len(train_truth),
                len(validation_truth),
            ):
                if representation == "pseudo_rgb":
                    raise AblationError(
                        "adapter pseudo_rgb existente nao corresponde aos "
                        "hiperparametros fixados"
                    )
                data_dir = args.data_dir.resolve() / representation
                prepare_command = [
                    sys.executable,
                    str(root / "07_prepare_mllm_finetuning.py"),
                    "--train",
                    str(args.train),
                    "--validation",
                    str(args.validation),
                    "--test-audit",
                    str(args.test),
                    "--representation",
                    representation,
                    "--output-dir",
                    str(data_dir),
                ]
                _run(prepare_command, root)
                training_command = [
                    sys.executable,
                    str(root / "08_finetune_mllm.py"),
                    "--model-id",
                    args.model_id,
                    "--train",
                    str(data_dir / "train.jsonl"),
                    "--validation",
                    str(data_dir / "validation.jsonl"),
                    "--representation",
                    representation,
                    "--output-dir",
                    str(adapter_dir),
                    "--environment-output",
                    str(environment_output),
                    "--epochs",
                    str(args.epochs),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--batch-size",
                    str(args.batch_size),
                    "--gradient-accumulation-steps",
                    str(args.gradient_accumulation_steps),
                    "--lora-r",
                    str(args.lora_r),
                    "--lora-alpha",
                    str(args.lora_alpha),
                    "--lora-dropout",
                    str(args.lora_dropout),
                    "--seed",
                    str(args.seed),
                    "--early-stopping-patience",
                    str(args.early_stopping_patience),
                    "--gradient-checkpointing"
                    if args.gradient_checkpointing
                    else "--no-gradient-checkpointing",
                    "--do-image-splitting"
                    if args.do_image_splitting
                    else "--no-do-image-splitting",
                ]
                if args.revision:
                    training_command.extend(["--revision", args.revision])
                if args.use_4bit:
                    training_command.append("--use-4bit")
                if args.trust_remote_code:
                    training_command.append("--trust-remote-code")
                if args.local_files_only:
                    training_command.append("--local-files-only")
                _run(training_command, root)
                if not _adapter_valid(
                    adapter_dir,
                    args,
                    representation,
                    len(train_truth),
                    len(validation_truth),
                ):
                    raise AblationError(f"adapter invalido para {representation}")
                actions[representation]["fine_tuning"] = "executed"
            else:
                actions[representation]["fine_tuning"] = "reused"

            if _evaluation_valid(
                finetuned_dir,
                args.validation,
                args.test,
                args.model_id,
                representation,
            ):
                actions[representation]["finetuned_inference"] = "reused"
            else:
                finetuned_command = _common_evaluation_command(
                    root / "09_evaluate_mllm_finetuned.py",
                    args,
                    representation,
                    finetuned_dir,
                    environment_output,
                )
                finetuned_command.extend(["--adapter-path", str(adapter_dir)])
                _run(finetuned_command, root)
                if not _evaluation_valid(
                    finetuned_dir,
                    args.validation,
                    args.test,
                    args.model_id,
                    representation,
                ):
                    raise AblationError(
                        f"saida fine-tuned invalida para {representation}"
                    )
                actions[representation]["finetuned_inference"] = "executed"

            _copy_prediction(
                zero_dir / "test_predictions.csv",
                results_dir / f"zero_shot_{representation}.csv",
            )
            _copy_prediction(
                finetuned_dir / "test_predictions.csv",
                results_dir / f"finetuned_{representation}.csv",
            )
            config = _read_json(adapter_dir / "training_config.json")
            adapter_configs[representation] = config
            actions[representation]["adapter_path"] = str(adapter_dir)

        _check_only_representation_changes(adapter_configs)
        manifest = {
            "status": "complete",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model_id,
            "revision": args.revision,
            "representations": args.representations,
            "actions": actions,
            "split_samples": {
                "train": len(train_truth),
                "validation": len(validation_truth),
                "test": len(test_truth),
            },
            "split_sha256": {
                "train": _sha256(args.train),
                "validation": _sha256(args.validation),
                "test": _sha256(args.test),
            },
            "prompt_sha256": hashlib.sha256(
                CLASSIFICATION_PROMPT.encode("utf-8")
            ).hexdigest(),
            "only_representation_changes": True,
        }
        _atomic_json(results_dir / "run_manifest.json", manifest)
    except Exception as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1

    print("\nAblation study concluida")
    for representation in args.representations:
        state = actions[representation]
        print(
            f"{representation}: zero-shot={state['zero_shot']}; "
            f"fine-tuning={state['fine_tuning']}; "
            f"inferencia={state['finetuned_inference']}"
        )
    print(f"Saida: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
