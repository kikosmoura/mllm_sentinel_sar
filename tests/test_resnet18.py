"""Regressoes de integridade cientifica; fixtures ficam somente em /tmp."""

from __future__ import annotations

import csv
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from cnn.data import (
    CNNError, DEFAULT_PREPROCESSING, SARDataset, audit_splits,
    load_examples, make_transform, resolve_png, verify_split_hashes,
)
from cnn.inference import export_predictions, infer, metric_rows
from cnn.model import build_pretrained
from cnn.runtime import read_json, write_json
from evaluation.metrics import Prediction, calculate_metrics


ROOT = Path(__file__).resolve().parents[1]


def module_from_path(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="resnet18-tests-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = {}
        for split, event in (("train", "Train"), ("validation", "Val"), ("test", "Test")):
            path = self.root / "data/splits" / f"{split}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.paths[split] = path
            fields = ["image_id", "event", "class", "class_id", "split",
                      "pseudo_rgb_image", "water_percentage"]
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for label in (0, 1):
                    image_id = f"{event}_{label}"
                    image = self.root / "data/images/pseudo_rgb" / f"{image_id}.png"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), (label * 180, 40, 70)).save(image)
                    writer.writerow({"image_id": image_id, "event": event,
                                     "class": "WATER" if label else "NON_WATER",
                                     "class_id": label, "split": split,
                                     "pseudo_rgb_image": str(image),
                                     "water_percentage": "MUST_NOT_BE_PARSED"})

    def test_audit_does_not_require_test_labels_or_images(self):
        path = self.paths["test"]
        path.write_text("image_id,event,split\nTest_0,Test,test\nTest_1,Test,test\n")
        for image in (self.root / "data/images/pseudo_rgb").glob("Test_*.png"):
            image.unlink()
        self.assertEqual(audit_splits(self.paths)["test"]["samples"], 2)

    def test_overlap_and_changed_csv_are_rejected(self):
        audit = audit_splits(self.paths)
        self.paths["test"].write_text("image_id,event,split\nTrain_0,Train,test\n")
        with self.assertRaisesRegex(CNNError, "compartilhados"):
            audit_splits(self.paths)
        with self.assertRaisesRegex(CNNError, "alterado"):
            verify_split_hashes(self.paths, audit)
        # Evento compartilhado tambem falha quando os IDs sao distintos.
        self.paths["test"].write_text("image_id,event,split\nTrain_99,Train,test\n")
        with self.assertRaisesRegex(CNNError, "events compartilhados"):
            audit_splits(self.paths)

    def test_old_path_copies_must_be_identical(self):
        local = self.root / "data/images/pseudo_rgb/Train_0.png"
        old = self.root / "legacy/data/images/pseudo_rgb/Train_0.png"
        old.parent.mkdir(parents=True)
        old.write_bytes(local.read_bytes())
        selected, record = resolve_png(str(old), "Train_0", "pseudo_rgb", self.root,
                                      self.paths["train"])
        self.assertEqual(selected, local)
        self.assertEqual(len(record["verified_copies"]), 2)
        old.write_bytes(b"different")
        with self.assertRaisesRegex(CNNError, "divergentes"):
            resolve_png(str(old), "Train_0", "pseudo_rgb", self.root, self.paths["train"])

    def test_corrupt_image_is_fatal(self):
        (self.root / "data/images/pseudo_rgb/Train_0.png").write_bytes(b"not a PNG")
        with self.assertRaisesRegex(CNNError, "leitura PNG"):
            load_examples(self.paths["train"], "train", "pseudo_rgb", self.root)

    def test_full_chip_transform_is_deterministic_and_retains_edges(self):
        image = Image.new("RGB", (512, 512), (0, 0, 0))
        # Uma borda branca desapareceria num center crop.
        for x in range(20):
            for y in range(512):
                image.putpixel((x, y), (255, 255, 255))
        transform = make_transform(DEFAULT_PREPROCESSING)
        first = transform(image)
        self.assertEqual(tuple(first.shape), (3, 224, 224))
        self.assertTrue(torch.equal(first, transform(image)))
        self.assertGreater(float(first[:, :, 0].mean()), float(first[:, :, 112].mean()))

    def test_invalid_logits_and_incomplete_coverage_are_fatal(self):
        examples, _ = load_examples(self.paths["validation"], "validation", "pseudo_rgb", self.root)
        loader = DataLoader(SARDataset(examples, DEFAULT_PREPROCESSING), batch_size=2)

        class Invalid(nn.Module):
            def forward(self, images):
                return torch.full((len(images), 2), float("nan"))

        with self.assertRaisesRegex(CNNError, "logits"):
            infer(Invalid(), loader, examples, torch.device("cpu"), "pseudo_rgb")
        with self.assertRaisesRegex(CNNError, "integralmente"):
            infer(Invalid(), [], examples, torch.device("cpu"), "pseudo_rgb")

    def test_probability_roundtrip_uses_shared_metrics(self):
        examples, _ = load_examples(self.paths["validation"], "validation", "pseudo_rgb", self.root)
        loader = DataLoader(SARDataset(examples, DEFAULT_PREPROCESSING), batch_size=2)

        class Fixed(nn.Module):
            def forward(self, images):
                return torch.tensor([[0.0, 0.1234567], [0.0, -0.7654321]])

        predicted, _ = infer(Fixed(), loader, examples, torch.device("cpu"), "pseudo_rgb")
        exported = export_predictions(self.root / "predictions.csv", predicted, examples, "pseudo_rgb")
        self.assertEqual(exported, predicted)
        for item in exported:
            self.assertLessEqual(abs(item.probability_non_water + item.probability_water - 1), 1e-9)
        rows = metric_rows(exported, "validation", 20, 42)
        shared = calculate_metrics(exported, "strict", bootstrap_samples=20, seed=42).to_dict()
        strict = next(row for row in rows if row["evaluation_view"] == "strict")
        for key, value in shared.items():
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(strict[key]), key)
            else:
                self.assertEqual(strict[key], value, key)

    def test_pretrained_download_failure_never_falls_back(self):
        with patch("cnn.model.resnet18", side_effect=RuntimeError("offline")) as constructor:
            with self.assertRaisesRegex(CNNError, "nenhum fallback"):
                build_pretrained(self.root / "cache")
        self.assertEqual(constructor.call_count, 1)

    def test_best_checkpoint_includes_warmup_and_stops_after_patience(self):
        trainer = module_from_path("train_selection_test", "12_train_resnet18.py")
        model = resnet18(weights=None)  # Somente fixture de teste, sem resultados reais.
        model.fc = nn.Linear(512, 2)
        args = trainer.parse_args(["--project-root", str(self.root), "--epochs", "8",
                                   "--warmup-epochs", "2", "--patience", "2",
                                   "--batch-size", "2", "--threads", "2"])
        losses = iter([0.6, 0.4, 0.4, 0.4])

        def controlled_validation(_model, _loader, examples, _device, representation, _criterion):
            predictions = [Prediction(
                image_id=item.truth.image_id, event=item.truth.event,
                true_class=item.truth.true_class, true_class_id=item.truth.true_class_id,
                model="ResNet-18", training="supervised-transfer-learning",
                representation=representation, predicted_class=item.truth.true_class,
                prediction_valid=True, raw_response="",
                probability_non_water=float(1 - item.truth.true_class_id),
                probability_water=float(item.truth.true_class_id),
            ) for item in examples]
            return predictions, next(losses)

        with patch.object(trainer, "build_pretrained", return_value=(model, {"fixture": True})), \
                patch.object(trainer, "infer", side_effect=controlled_validation):
            config = trainer.train(args)
        self.assertEqual(config["stop_reason"], "early_stopping")
        self.assertEqual(config["epochs_executed"], 4)
        self.assertEqual(config["best_epoch"], 2)
        self.assertEqual(config["best_phase"], "head-only")
        self.assertEqual(config["best_validation_loss"], 0.4)
        checkpoint = torch.load(args.model_dir / "best_checkpoint.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["phase"], "head-only")

    def test_training_does_not_open_test_images_or_labels(self):
        trainer = module_from_path("train_resnet18_test", "12_train_resnet18.py")
        evaluator = module_from_path("evaluate_resnet18_test", "13_evaluate_resnet18.py")
        self.paths["test"].write_text("image_id,event,split\nTest_0,Test,test\nTest_1,Test,test\n")
        for image in (self.root / "data/images/pseudo_rgb").glob("Test_*.png"):
            image.unlink()
        audit_path = self.root / "preflight_audit.json"
        write_json(audit_path, audit_splits(self.paths))
        # Fixture isolada de arquitetura, explicitamente sem pesos ImageNet.
        # O smoke test separado usa os pesos oficiais e dados reais.
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        origin = {"enum": "ResNet18_Weights.IMAGENET1K_V1", "fixture": True}
        args = trainer.parse_args(["--project-root", str(self.root), "--epochs", "2",
                                   "--warmup-epochs", "1", "--batch-size", "2", "--threads", "2",
                                   "--split-audit", str(audit_path)])
        original_open = Path.open

        def guarded_open(path, *open_args, **kwargs):
            if path.resolve() == self.paths["test"].resolve():
                raise AssertionError("test.csv nao pode ser aberto durante treinamento")
            return original_open(path, *open_args, **kwargs)

        with patch.object(trainer, "build_pretrained", return_value=(model, origin)), \
                patch.object(Path, "open", guarded_open):
            config = trainer.train(args)
        self.assertEqual(config["test_samples_used_for_training"], 0)
        self.assertEqual(set(config["image_resolutions"]), {"train", "validation"})
        with (args.results_dir / "training_history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["phase"] for row in rows], ["head-only", "layer4+fc"])
        # Avaliar somente validation; ate aqui o teste continua sem rotulos/imagens.
        eval_args = evaluator.parse_args(["--project-root", str(self.root),
                                          "--checkpoint", str(args.model_dir / "best_checkpoint.pt"),
                                          "--splits", "validation", "--threads", "2",
                                          "--bootstrap-samples", "20"])
        report = evaluator.evaluate(eval_args)
        self.assertEqual(report["samples"], {"validation": 2})
        self.assertFalse((args.results_dir / "test_predictions.csv").exists())
        self.assertEqual(read_json(args.model_dir / "training_config.json")["status"], "complete")
        with self.assertRaisesRegex(CNNError, "avaliacao existente"):
            evaluator.evaluate(eval_args)
        # Qualquer mudanca no split depois do treino impede avaliar ate validation.
        self.paths["test"].write_text(self.paths["test"].read_text() + "Test_2,Test,test\n")
        with self.assertRaisesRegex(CNNError, "splits diferentes"):
            evaluator.evaluate(eval_args)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
