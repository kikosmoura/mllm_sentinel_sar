"""Verificacao curta com pesos oficiais, sem treino/artefatos experimentais."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn.data import (
    DEFAULT_PREPROCESSING, SARDataset, audit_splits, load_examples,
    verify_split_hashes,
)
from cnn.inference import export_predictions, infer, metric_rows
from cnn.model import build_pretrained, make_optimizer, restore_model, set_training_phase
from cnn.runtime import save_checkpoint, seed_everything, write_csv, write_json


def verify(project_root: Path, weights_cache: Path | None = None):
    torch.set_num_threads(2)
    seed_everything(42)
    paths = {split: project_root / "data/splits" / f"{split}.csv"
             for split in ("train", "validation", "test")}
    audit = audit_splits(paths)
    train, _ = load_examples(paths["train"], "train", "pseudo_rgb", project_root)
    validation, _ = load_examples(paths["validation"], "validation", "pseudo_rgb", project_root)
    # Somente duas imagens de cada split para os passos computacionais.
    small_train = [next(item for item in train if item.truth.true_class_id == cls)
                   for cls in (0, 1)]
    small_validation = [next(item for item in validation if item.truth.true_class_id == cls)
                        for cls in (0, 1)]
    with tempfile.TemporaryDirectory(prefix="resnet18-smoke-") as directory:
        output = Path(directory)
        model, origin = build_pretrained(weights_cache or output / "weights")
        loader = DataLoader(SARDataset(small_train, DEFAULT_PREPROCESSING, training=True),
                            batch_size=2, shuffle=False)
        images, targets, _ = next(iter(loader))
        criterion = nn.CrossEntropyLoss()
        phase_counts = {}
        for phase in ("head-only", "layer4+fc"):
            phase_counts[phase] = set_training_phase(model, phase)
            before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            optimizer = make_optimizer(model, 1e-3, 1e-4, 1e-4)
            logits = model(images)
            assert logits.shape == (2, 2)
            loss = criterion(logits, targets)
            assert bool(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
                else:
                    assert parameter.grad is None, name
            optimizer.step()
            after = model.state_dict()
            assert not torch.equal(before["fc.weight"], after["fc.weight"])
            for name, old in before.items():
                if "running_" in name or "num_batches_tracked" in name:
                    assert torch.equal(old, after[name]), name
                if not name.startswith("fc.") and not (phase == "layer4+fc" and name.startswith("layer4.")):
                    assert torch.equal(old, after[name]), name
            if phase == "layer4+fc":
                assert not torch.equal(before["layer4.0.conv1.weight"], after["layer4.0.conv1.weight"])
        checkpoint = {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "config": {"preprocessing": DEFAULT_PREPROCESSING, "weights_origin": origin},
        }
        save_checkpoint(output / "smoke_checkpoint.pt", checkpoint)
        restored = restore_model(torch.load(output / "smoke_checkpoint.pt",
                                            weights_only=True, map_location="cpu"))
        model.eval()
        with torch.inference_mode():
            assert torch.equal(model(images), restored(images))
        val_loader = DataLoader(SARDataset(small_validation, DEFAULT_PREPROCESSING),
                                batch_size=2, shuffle=False)
        predictions, _ = infer(restored, val_loader, small_validation,
                               torch.device("cpu"), "pseudo_rgb")
        exported = export_predictions(output / "smoke_predictions.csv", predictions,
                                      small_validation, "pseudo_rgb")
        metrics = metric_rows(exported, "validation", bootstrap_samples=20, seed=42)
        write_csv(output / "smoke_metrics.csv", metrics)
        write_json(output / "smoke_config.json", checkpoint["config"])
        assert len(exported) == 2
        for item in exported:
            assert abs(item.probability_non_water + item.probability_water - 1) <= 1e-9
        # Integra tambem a escrita de matriz de confusao usada pela CLI.
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluate_resnet18", project_root /
                                                     "13_evaluate_resnet18.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.save_confusion(exported, output / "smoke_confusion.png", "Smoke test")
        assert (output / "smoke_confusion.png").stat().st_size > 0
        verify_split_hashes(paths, audit)
        print("OK: pesos IMAGENET1K_V1; forward/backward nas duas fases; "
              "blocos e BatchNorm congelados; checkpoint restaurado; "
              "CSV validado; metricas/ICs e matriz de confusao exportados.")
        print(f"Parametros: {phase_counts}")
        print("Somente 2 passos de otimizacao e 2 imagens de validacao; "
              "imagens/rotulos de teste nao carregados. Artefatos temporarios removidos.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--weights-cache", type=Path)
    args = parser.parse_args()
    verify(args.project_root.resolve(), args.weights_cache)


if __name__ == "__main__":
    main()
