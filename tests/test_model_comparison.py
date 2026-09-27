"""Regressoes da consolidacao; nenhum modelo e treinado/carregado."""

from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evaluation.comparison import (
    CNN_MODEL, display_name, main_model_names, sha256, validate_cnn_artifacts,
)
from evaluation.metrics import EvaluationError, Prediction


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_results_tests", ROOT / "11_generate_final_results.py")
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class CNNProvenanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="comparison-tests-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.paths = {}
        for name in ("train", "validation", "test"):
            path = self.root / f"{name}.csv"
            path.write_text(f"fixture {name}\n")
            self.paths[name] = path
        self.config_path = self.root / "training_config.json"
        checkpoint = self.root / "best_checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint fixture; never deserialized")
        (self.root / "test_predictions.csv").write_text("fixture predictions\n")
        self.config = {
            "status": "complete", "model": CNN_MODEL, "representation": "pseudo_rgb",
            "input_source": "SAR PNG pixels only", "class_to_id": {"NON_WATER": 0, "WATER": 1},
            "test_samples_used_for_training": 0, "test_samples_used_for_selection": 0,
            "weights_origin": {"enum": "ResNet18_Weights.IMAGENET1K_V1"},
            "split_sha256": {name: sha256(path) for name, path in self.paths.items()},
            "preprocessing": {"resize": "full-chip", "size": [224, 224]},
            "parameter_counts_by_phase": {"layer4+fc": {"total_parameters": 10,
                                                         "trainable_parameters": 4}},
            "best_phase": "layer4+fc", "best_checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "seed": 42,
        }
        self.evaluation = {
            "status": "complete", "checkpoint_sha256": self.config["checkpoint_sha256"],
            "split_sha256": self.config["split_sha256"], "representation": "pseudo_rgb",
            "preprocessing": self.config["preprocessing"], "seed": 42,
            "test_samples_used_for_training": 0, "test_samples_used_for_selection": 0,
        }
        self.save()

    def save(self):
        self.config_path.write_text(json.dumps(self.config))
        (self.root / "evaluation_manifest.json").write_text(json.dumps(self.evaluation))

    def test_configuration_and_prediction_provenance(self):
        _, validated = validate_cnn_artifacts(self.config_path, self.root, self.paths)
        self.assertTrue(validated["cnn_split_hashes_match"])
        self.assertEqual(validated["cnn_test_predictions_sha256"], sha256(self.root / "test_predictions.csv"))

    def test_changed_split_is_rejected(self):
        self.paths["test"].write_text("changed split")
        with self.assertRaisesRegex(EvaluationError, "hashes dos splits"):
            validate_cnn_artifacts(self.config_path, self.root, self.paths)

    def test_test_usage_and_wrong_representation_are_rejected(self):
        for key, value, message in (("test_samples_used_for_training", 1, "deve ser zero"),
                                    ("test_samples_used_for_selection", 1, "deve ser zero"),
                                    ("representation", "vv", "pseudo_rgb")):
            with self.subTest(key=key):
                old = self.config[key]
                self.config[key] = value
                self.save()
                with self.assertRaisesRegex(EvaluationError, message):
                    validate_cnn_artifacts(self.config_path, self.root, self.paths)
                self.config[key] = old

    def test_changed_checkpoint_or_mismatched_evaluation_is_rejected(self):
        self.evaluation["checkpoint_sha256"] = "different"
        self.save()
        with self.assertRaisesRegex(EvaluationError, "avaliacao nao corresponde"):
            validate_cnn_artifacts(self.config_path, self.root, self.paths)
        (self.root / "best_checkpoint.pt").write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(EvaluationError, "hash do checkpoint"):
            validate_cnn_artifacts(self.config_path, self.root, self.paths)


class ConsolidationTests(unittest.TestCase):
    def test_main_model_sets_and_display_names(self):
        self.assertEqual(len(main_model_names(False)), 3)
        self.assertEqual(len(main_model_names(True)), 4)
        self.assertEqual(display_name("MLLM Fine-Tuned"), "MLLM + LoRA")

    def test_views_are_required_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory(prefix="comparison-views-") as directory:
            path = Path(directory) / "comparison.csv"
            rows = [{"model": model, "evaluation_view": view,
                     "representation": "vv_vh_features" if model == "Random Forest" else "pseudo_rgb"}
                    for model in main_model_names(True) for view in ("strict", "valid-only")]

            def save(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(values)

            save(rows)
            self.assertEqual(len(final._load_main_strict(path, True)), 4)
            save(rows + [rows[0]])
            with self.assertRaises(final.FinalResultsError):
                final._load_main_strict(path, True)
            save([row for row in rows if row["model"] != CNN_MODEL])
            self.assertEqual(len(final._load_main_strict(path, False)), 3)

    def test_consolidation_includes_cnn_once_and_detects_duplicate_mllm(self):
        sample = Prediction("Event_1", "Event", "WATER", 1, "Random Forest",
                            "supervised", "vv_vh_features", "WATER", True, "", 0.1, 0.9)
        cnn = replace(sample, model=CNN_MODEL, training="supervised-transfer-learning",
                      representation="pseudo_rgb")
        ablation = {(training, rep): [replace(sample, model="MLLM", training=training,
                                              representation=rep)]
                    for training in ("zero-shot", "fine-tuned")
                    for rep in ("vv", "vh", "pseudo_rgb")}
        base = {"Random Forest": [sample], CNN_MODEL: [cnn]}
        self.assertEqual(len(final._consolidated_rows(base, ablation)), 8)
        base["MLLM Zero-Shot"] = [replace(ablation[("zero-shot", "pseudo_rgb")][0],
                                         model="MLLM Zero-Shot")]
        with self.assertRaisesRegex(final.FinalResultsError, "duplicada"):
            final._consolidated_rows(base, ablation)

    def test_grouped_bars_are_centered_for_three_or_four_models(self):
        for count in (3, 4):
            rows = [{"model": name, **{key: "0.5" for key in
                    ("accuracy", "balanced_accuracy", "precision", "recall", "f1")}}
                    for name in main_model_names(count == 4)]

            def inspect(figure, _path):
                axis = figure.axes[0]
                centers = [container.patches[0].get_x() + container.patches[0].get_width() / 2
                           for container in axis.containers]
                self.assertEqual(len(centers), count)
                self.assertAlmostEqual(sum(centers) / len(centers), 0)
                self.assertLessEqual(sum(c.patches[0].get_width() for c in axis.containers), 0.8 + 1e-12)
                final.plt.close(figure)

            with patch.object(final, "_save_figure", side_effect=inspect):
                final._main_metrics_figure(rows, Path("unused.png"))


if __name__ == "__main__":
    unittest.main()
