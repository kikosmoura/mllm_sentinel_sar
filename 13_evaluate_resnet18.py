#!/usr/bin/env python3
"""Avalia separadamente o checkpoint ResNet-18 selecionado na validacao."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from cnn.data import (
    CNNError, SARDataset, audit_splits, file_sha256, load_examples,
    verify_image_hashes, verify_split_hashes,
)
from cnn.inference import export_predictions, infer, metric_rows
from cnn.model import restore_model
from cnn.runtime import (
    environment, read_json, seed_everything, seed_worker, select_device,
    write_csv, write_json,
)
from evaluation.metrics import CLASS_TO_ID, EvaluationError, confusion_counts


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path,
                        help="Padrao models/resnet18_pseudo_rgb/best_checkpoint.pt na raiz")
    parser.add_argument("--training-config", type=Path,
                        help="Padrao training_config.json junto ao checkpoint")
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"),
                        default=["validation", "test"])
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    for split in ("train", "validation", "test"):
        if getattr(args, split) is None:
            setattr(args, split, args.project_root / "data/splits" / f"{split}.csv")
    args.checkpoint = (args.checkpoint or args.project_root /
                       "models/resnet18_pseudo_rgb/best_checkpoint.pt").resolve()
    args.training_config = (args.training_config or args.checkpoint.parent /
                            "training_config.json").resolve()
    if args.batch_size <= 0 or args.threads <= 0 or args.num_workers < 0 or args.bootstrap_samples < 0:
        parser.error("batch-size/threads > 0; num-workers/bootstrap-samples >= 0")
    if len(set(args.splits)) != len(args.splits):
        parser.error("--splits nao pode conter duplicatas")
    return args


def save_confusion(predictions, path: Path, title: str):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/resnet18_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    try:
        ConfusionMatrixDisplay(confusion_counts(predictions),
                               display_labels=["NON_WATER", "WATER"]).plot(
                                   ax=axis, cmap="Blues", colorbar=False, values_format="d")
        axis.set_title(title)
        figure.tight_layout()
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            figure.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        plt.close(figure)


def evaluate(args):
    torch.set_num_threads(args.threads)
    metadata = read_json(args.training_config)
    if metadata.get("status") != "complete" or metadata.get("test_samples_used_for_training") != 0:
        raise CNNError("treinamento incompleto ou uso de teste nao comprovadamente zero")
    if metadata.get("test_samples_used_for_selection") != 0:
        raise CNNError("teste foi usado na selecao")
    if file_sha256(args.checkpoint) != metadata.get("checkpoint_sha256"):
        raise CNNError("hash do checkpoint diverge da configuracao")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    # JSON externo e checkpoint devem descrever exatamente o mesmo treinamento.
    if {key: value for key, value in metadata.items() if key != "checkpoint_sha256"} != config:
        raise CNNError("configuracao externa diverge do checkpoint")
    if config.get("class_to_id") != CLASS_TO_ID:
        raise CNNError("mapeamento de classes divergente")
    if config.get("weights_origin", {}).get("enum") != "ResNet18_Weights.IMAGENET1K_V1":
        raise CNNError("origem ImageNet dos pesos divergente")
    paths = {split: getattr(args, split).resolve()
             for split in ("train", "validation", "test")}
    audit = audit_splits(paths)
    hashes = {split: data["sha256"] for split, data in audit.items()}
    if hashes != config["split_sha256"]:
        raise CNNError("splits diferentes daqueles registrados no treinamento")
    seed_everything(config["seed"])
    device = select_device(args.device)
    model = restore_model(checkpoint).to(device)
    representation = config["representation"]
    output = (args.results_dir or args.project_root / "results" /
              f"resnet18_{representation}").resolve()
    protected = [output / "metrics.csv", output / "evaluation_manifest.json"]
    protected.extend(output / f"{split}_predictions.csv" for split in args.splits)
    if any(path.exists() for path in protected):
        raise CNNError(f"avaliacao existente; escolha outro --results-dir: {output}")
    all_metrics, runs, image_resolutions = [], {}, {}
    # Calcular tudo antes de escrever resultados, para falhas de leitura nao
    # deixarem um conjunto incompleto aparentemente oficial.
    for split in args.splits:
        examples, resolutions = load_examples(paths[split], split, representation, args.project_root)
        loader = DataLoader(
            SARDataset(examples, config["preprocessing"]), batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config["seed"]))
        started = time.perf_counter()
        predictions, _ = infer(model, loader, examples, device, representation)
        elapsed = time.perf_counter() - started
        verify_image_hashes(examples)
        runs[split] = (predictions, examples, elapsed)
        image_resolutions[split] = resolutions
    verify_split_hashes(paths, audit)
    output.mkdir(parents=True, exist_ok=True)
    for split, (predictions, examples, elapsed) in runs.items():
        validated = export_predictions(output / f"{split}_predictions.csv",
                                       predictions, examples, representation)
        metrics = metric_rows(validated, split, args.bootstrap_samples, config["seed"])
        all_metrics.extend(metrics)
        save_confusion(validated, output / f"confusion_matrix_{split}.png",
                       f"ResNet-18 - {representation} - {split}")
        strict = next(row for row in metrics if row["evaluation_view"] == "strict")
        print(f"{split}: samples={len(validated)} accuracy={strict['accuracy']:.6f} "
              f"balanced_accuracy={strict['balanced_accuracy']:.6f} "
              f"f1={strict['f1']:.6f} seconds={elapsed:.2f}", flush=True)
    write_csv(output / "metrics.csv", all_metrics)
    manifest = {
        "status": "complete", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_config": str(args.training_config), "representation": representation,
        "preprocessing": config["preprocessing"], "weights_origin": config["weights_origin"],
        "split_sha256": hashes, "geographic_isolation": True,
        "image_resolutions": image_resolutions,
        "samples": {split: len(values[0]) for split, values in runs.items()},
        "inference_seconds": {split: values[2] for split, values in runs.items()},
        "bootstrap_samples": args.bootstrap_samples, "seed": config["seed"],
        "decision": "argmax; no threshold tuning on test",
        "test_samples_used_for_training": 0, "test_samples_used_for_selection": 0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **environment(device),
    }
    write_json(output / "evaluation_manifest.json", manifest)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    try:
        evaluate(args)
    except (CNNError, EvaluationError, OSError, RuntimeError, KeyError) as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
