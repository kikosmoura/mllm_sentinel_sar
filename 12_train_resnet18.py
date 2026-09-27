#!/usr/bin/env python3
"""Treina ResNet-18 usando somente pixels/rotulos de train e validation."""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Necessario para operacoes CUDA deterministicas, antes de inicializar CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn.data import (
    CNNError, DEFAULT_PREPROCESSING, REPRESENTATIONS, SARDataset, audit_splits,
    file_sha256, load_examples, validate_split_audit, verify_image_hashes, verify_split_hashes,
)
from cnn.inference import infer
from cnn.model import MODES, build_pretrained, make_optimizer, set_training_phase
from cnn.runtime import (
    environment, read_json, save_checkpoint, seed_everything, seed_worker, select_device,
    write_csv, write_json,
)
from evaluation.metrics import CLASS_TO_ID, EvaluationError, calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}", type=Path,
                            help="CSV; test e lido somente para IDs/eventos e hash")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="pseudo_rgb")
    parser.add_argument("--mode", choices=MODES, default="layer4+fc")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--weights-cache", type=Path,
                        help="Cache torch.hub; padrao .cache/resnet18 na raiz do projeto")
    parser.add_argument("--split-audit", type=Path,
                        help="Auditoria pre-execucao em JSON; evita abrir test.csv durante o treino")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25, help="Maximo total, incluindo aquecimento")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-fc", type=float, default=1e-3)
    parser.add_argument("--lr-layer4", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    for split in ("train", "validation", "test"):
        if getattr(args, split) is None:
            setattr(args, split, args.project_root / "data/splits" / f"{split}.csv")
    args.model_dir = (args.model_dir or args.project_root / "models" /
                      f"resnet18_{args.representation}").resolve()
    args.results_dir = (args.results_dir or args.project_root / "results" /
                        f"resnet18_{args.representation}").resolve()
    args.weights_cache = (args.weights_cache or args.project_root /
                          ".cache/resnet18").resolve()
    for key in ("batch_size", "epochs", "patience", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} deve ser positivo")
    if args.num_workers < 0 or args.seed < 0 or args.seed >= 2**32:
        parser.error("num-workers >= 0 e seed em [0, 2**32)")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("warmup-epochs deve estar entre 0 e epochs")
    if args.mode == "layer4+fc" and args.warmup_epochs >= args.epochs:
        parser.error("layer4+fc exige ao menos uma epoca apos o aquecimento")
    for key in ("lr_fc", "lr_layer4", "weight_decay"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            parser.error(f"--{key.replace('_', '-')} invalido")
    if (args.model_dir == args.results_dir or args.model_dir in args.results_dir.parents
            or args.results_dir in args.model_dir.parents):
        parser.error("model-dir e results-dir devem ser separados")
    return args


def training_epoch(model, loader, optimizer, criterion, device):
    loss_sum = weight_sum = 0.0
    for images, targets, _indices in loader:
        # Apenas pixels sao passados ao modelo.
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise CNNError("logits de treino nao finitos")
        losses = criterion(logits, targets)
        denominator = criterion.weight[targets].sum()
        loss = losses.sum() / denominator
        if not bool(torch.isfinite(loss)):
            raise CNNError("loss de treino nao finita")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise CNNError(f"gradiente nao finito: {name}")
        optimizer.step()
        loss_sum += float(losses.detach().sum())
        weight_sum += float(denominator)
    return loss_sum / weight_sum


def train(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    paths = {split: getattr(args, split).resolve()
             for split in ("train", "validation", "test")}
    if args.split_audit is None:
        audit = audit_splits(paths)
        checked_paths = paths
    else:
        audit = read_json(args.split_audit)
        validate_split_audit(audit)
        for name, path in paths.items():
            if audit[name]["path"] != str(path):
                raise CNNError(f"caminho diverge da auditoria pre-execucao: {name}")
        # O teste foi auditado ANTES do treinamento. Reconfirmar seu hash
        # somente na avaliacao, depois de fixar o checkpoint.
        checked_paths = {name: path for name, path in paths.items() if name != "test"}
    # NAO carregar imagens/rotulos de test. Somente estes dois datasets existem.
    examples, resolutions = {}, {}
    for split in ("train", "validation"):
        examples[split], resolutions[split] = load_examples(
            paths[split], split, args.representation, args.project_root)
    verify_split_hashes(checked_paths, audit)
    for directory in (args.model_dir, args.results_dir):
        if directory.exists() and any(directory.iterdir()):
            raise CNNError(f"diretorio de saida nao vazio; escolha outro: {directory}")
    model, weights = build_pretrained(args.weights_cache)
    model.to(device)
    counts = Counter(item.truth.true_class_id for item in examples["train"])
    class_weights = [len(examples["train"]) / (2 * counts[label]) for label in (0, 1)]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        reduction="none")
    preprocessing = copy.deepcopy(DEFAULT_PREPROCESSING)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SARDataset(examples["train"], preprocessing, training=True),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=generator)
    validation_loader = DataLoader(
        SARDataset(examples["validation"], preprocessing), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + 1))
    phase_parameters = {mode: set_training_phase(model, mode) for mode in MODES}
    config = {
        "format_version": 1, "status": "running", "model": "ResNet-18",
        "training": "supervised-transfer-learning", "input_source": "SAR PNG pixels only",
        "class_to_id": CLASS_TO_ID, "representation": args.representation,
        "preprocessing": preprocessing, "weights_origin": weights,
        "mode": args.mode, "warmup_epochs": args.warmup_epochs,
        "max_epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "optimizer": "AdamW",
        "lr_fc": args.lr_fc, "lr_layer4": args.lr_layer4,
        "weight_decay": args.weight_decay,
        "optimizer_reset_at_phase_transition": True,
        "batchnorm_running_statistics": "frozen in every phase",
        "parameter_counts_by_phase": phase_parameters,
        "class_counts_train": {str(key): counts[key] for key in (0, 1)},
        "class_weights_train": class_weights,
        "loss": "weighted cross entropy; sum(loss)/sum(target class weights)",
        "selection": "validation balanced_accuracy descending, then loss ascending",
        "patience": args.patience, "early_stopping": "after warmup only",
        "split_sha256": {name: item["sha256"] for name, item in audit.items()},
        "split_audit": audit, "image_resolutions": resolutions,
        "test_access": ("pre-execution audit metadata only; test CSV/images/labels not opened"
                        if args.split_audit is not None
                        else "opaque CSV hash and ID/event/split audit only"),
        "split_audit_source": ({"path": str(args.split_audit.resolve()),
                                "sha256": file_sha256(args.split_audit)}
                               if args.split_audit is not None else None),
        "test_samples_used_for_training": 0, "test_samples_used_for_selection": 0,
        "training_samples": len(examples["train"]),
        "validation_samples": len(examples["validation"]),
        "num_workers": args.num_workers,
        "deterministic_algorithms": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **environment(device),
    }
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.model_dir / "training_config.json", config)
    checkpoint_path = args.model_dir / "best_checkpoint.pt"
    history = []
    best_key = (-math.inf, -math.inf)
    bad_epochs = 0
    phase = None
    started = time.perf_counter()
    stop_reason = "max_epochs"
    for epoch in range(1, args.epochs + 1):
        next_phase = ("head-only" if args.mode == "head-only" or epoch <= args.warmup_epochs
                      else "layer4+fc")
        if next_phase != phase:
            phase = next_phase
            # Reset explicito de AdamW na transicao; taxas distintas por grupo.
            set_training_phase(model, phase)
            optimizer = make_optimizer(model, args.lr_fc, args.lr_layer4, args.weight_decay)
        set_training_phase(model, phase)  # infer() colocou tudo em eval na epoca anterior.
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        train_loss = training_epoch(model, train_loader, optimizer, criterion, device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        predictions, validation_loss = infer(
            model, validation_loader, examples["validation"], device,
            args.representation, criterion)
        validation_seconds = time.perf_counter() - validation_started
        metric = calculate_metrics(predictions, "strict", bootstrap_samples=0, seed=args.seed)
        score = metric.balanced_accuracy
        if not math.isfinite(score) or not math.isfinite(validation_loss):
            raise CNNError("criterio de selecao nao finito")
        key = (score, -validation_loss)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch, best_phase, best_loss = epoch, phase, validation_loss
            save_checkpoint(checkpoint_path, {
                "format_version": 1,
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()},
                "epoch": epoch, "phase": phase,
                "validation_balanced_accuracy": score,
                "validation_loss": validation_loss,
                "config": config,
            })
        if epoch > args.warmup_epochs:
            bad_epochs = 0 if improved else bad_epochs + 1
        history.append({
            "epoch": epoch, "phase": phase, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": metric.accuracy,
            "validation_balanced_accuracy": score,
            "validation_precision": metric.precision,
            "validation_recall": metric.recall,
            "validation_f1": metric.f1, "validation_roc_auc": metric.roc_auc,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "trainable_parameters": phase_parameters[phase]["trainable_parameters"],
            "checkpoint_selected": improved, "bad_epochs_after_warmup": bad_epochs,
        })
        write_csv(args.results_dir / "training_history.csv", history)
        print(f"epoch={epoch}/{args.epochs} phase={phase} train_loss={train_loss:.6f} "
              f"validation_loss={validation_loss:.6f} balanced_accuracy={score:.6f} "
              f"best={best_epoch} seconds={history[-1]['epoch_seconds']:.2f}", flush=True)
        if epoch > args.warmup_epochs and bad_epochs >= args.patience:
            stop_reason = "early_stopping"
            break
    verify_split_hashes(checked_paths, audit)
    for split in examples:
        verify_image_hashes(examples[split])
    config.update({
        "status": "complete", "epochs_executed": len(history), "stop_reason": stop_reason,
        "best_epoch": best_epoch, "best_phase": best_phase,
        "best_validation_balanced_accuracy": best_key[0], "best_validation_loss": best_loss,
        "best_checkpoint": str(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    selected = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selected["config"] = config
    save_checkpoint(checkpoint_path, selected)
    config["checkpoint_sha256"] = file_sha256(checkpoint_path)
    write_json(args.model_dir / "training_config.json", config)
    write_json(args.results_dir / "training_summary.json", config)
    print(f"Checkpoint: {checkpoint_path}\nMelhor epoca: {best_epoch}; "
          f"balanced_accuracy validation={best_key[0]:.6f}\n"
          "Teste nao avaliado. Use 13_evaluate_resnet18.py para inferencia.")
    return config


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except (CNNError, EvaluationError, OSError, RuntimeError) as exc:
        print(f"[ERRO FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
