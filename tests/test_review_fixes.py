"""Independent boundary cases raised in the external reproduction review."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import h5py
import numpy as np
from repro import common
from repro.cache_compat import load_index, read_candidate
from repro.inference_provenance import Inventory, selection_provenance
from screening.croco_native import save_cache, RAW_SIGNALS


class ReviewFixTests(unittest.TestCase):
    def test_output_uses_configured_mount_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "other-user-volume"
            root.mkdir()
            with patch.object(
                common, "profile", return_value={"storage_root": str(root)}
            ):
                self.assertEqual(common.output("valid"), root / "valid")
                (root / "escape").symlink_to(Path(tmp), target_is_directory=True)
                with self.assertRaises(ValueError):
                    common.output("escape")
            with patch.object(
                common,
                "profile",
                return_value={
                    "storage_root": str(root),
                    "storage_guard_root": str(root / "guard"),
                },
            ):
                with self.assertRaises(ValueError):
                    common.output("outside")
            from sensor_v4_common import storage

            self.assertEqual(storage({"storage_root": str(root)}, "x"), root / "x")
            with self.assertRaises(ValueError):
                storage({"storage_root": str(root)}, "../escape")

    def test_doctor_marks_missing_distribution(self):
        from repro.cli import package_version
        from importlib.metadata import PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            self.assertEqual(package_version("not-installed"), "missing")

    def test_phase_a_missing_assets_fails_before_model(self):
        from repro.paper import require_assets

        with self.assertRaises(FileNotFoundError):
            require_assets(
                [{"assets": {"left": {"path": "__missing_review_fixture__/x.png"}}}]
            )

    def test_complete_vendor_snapshot(self):
        from repro.vendor import verify_vendor

        self.assertEqual(verify_vendor()["files"], 203)

    def test_broader_recipe_import_and_determinism(self):
        from corruptions_robustspring_v2 import corrupt

        rgb = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
        for name in ("gaussian_blur", "jpeg_compression", "pixelate"):
            first = corrupt(rgb, name, 12)
            self.assertEqual(first.shape, rgb.shape)
            np.testing.assert_array_equal(first, corrupt(rgb, name, 12))

    def fixture(self, folder):
        row = {"condition": "clean", "pair_id": "s/0001", "scene": "s", "frame": "0001"}
        inference = {
            "checkpoint_sha256": "checkpoint",
            "upstream_commit": "upstream",
            "overlap": 0.7,
            "precision": "fp16",
            "precision_scope": "forward-only",
            "GPU": "GPU-a",
            "batch": 2,
        }
        source = folder / "run.json"
        common.write(source, {"execution": "unit-fixture", "inference": inference})
        attest = {
            "schema": "legacy_cache_attestation_v1",
            "cache_match": {
                "config_sha256": "config",
                "checkpoint_sha256": "checkpoint",
                "condition": "clean",
            },
            "inference": inference,
            "sources": [{"path": str(source), "sha256": common.sha(source)}],
            "rationale": "synthetic fixture for strict read-only compatibility checks",
        }
        att = folder / "att.json"
        common.write(att, attest)
        common.write(
            folder / "CACHE_INDEX.json",
            {
                "identity": {
                    "root": str(folder),
                    "layout": "legacy_v1",
                    "evidence": {"path": str(att), "sha256": common.sha(att)},
                }
            },
        )
        arrays = {
            k: np.ones((2, 3, 8), np.float32) for k in ["disparity", *RAW_SIGNALS]
        }
        meta = {
            **row,
            "candidate_id": "identity",
            "config_sha256": "config",
            "checkpoint_sha256": "checkpoint",
            "precision": "FP16",
            "effective_reference_batch_size": 2,
        }
        path = folder / "cache/clean/identity/s/0001.h5"
        save_cache(path, arrays, meta)
        return row, inference, path, source

    def test_legacy_read_is_lossless_and_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row, expected, path, _ = self.fixture(folder)
            before = common.sha(path)
            entry = load_index(folder, ["identity"])["identity"]
            arrays, meta, receipt = read_candidate(
                entry, row, "identity", RAW_SIGNALS, shape=(2, 3, 8)
            )
            self.assertEqual({k: meta[k] for k in expected}, expected)
            np.testing.assert_array_equal(
                arrays["disparity"], np.ones((2, 3, 8), np.float32)
            )
            self.assertIn("precision_scope", receipt["supplemented_fields"])
            self.assertEqual(before, common.sha(path))
            # The same reader must retain the native cache path and metadata.
            native = folder / "cache/clean/s/0001_identity.h5"
            save_cache(native, arrays, meta)
            _, other, _ = read_candidate(
                {"root": str(folder)}, row, "identity", RAW_SIGNALS, shape=(2, 3, 8)
            )
            self.assertEqual(meta, other)

    def test_legacy_missing_tampered_or_conflicting_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            row, _, path, source = self.fixture(folder)
            entry = load_index(folder, ["identity"])["identity"]
            entry["attestation"]["inference"]["precision"] = "fp32"
            with self.assertRaises(ValueError):
                read_candidate(entry, row, "identity", RAW_SIGNALS, shape=(2, 3, 8))
            source.write_text("{}")
            with self.assertRaises(ValueError):
                load_index(folder, ["identity"])
            common.write(
                folder / "CACHE_INDEX.json",
                {"identity": {"root": str(folder), "layout": "legacy_v1"}},
            )
            with self.assertRaises(ValueError):
                load_index(folder, ["identity"])

    def test_precision_scope_difference_and_unknowns_are_visible(self):
        a = {
            "checkpoint_sha256": "c",
            "upstream_commit": "u",
            "overlap": 0.7,
            "precision": "fp16",
            "precision_scope": "forward-only",
            "GPU": "A",
            "batch": 2,
            "candidate_id": "identity",
        }
        inv = Inventory()
        inv.add("clean", {"inference_provenance": selection_provenance([a], [])})
        inv.add(
            "noise",
            {
                "inference_provenance": selection_provenance(
                    [{**a, "precision_scope": "whole-call", "GPU": "B"}], []
                )
            },
        )
        q = inv.summary()
        self.assertTrue(q["mixed_precision"])
        self.assertTrue(q["mixed_precision_scope"])
        self.assertFalse(q["mixed_precision_dtype"])
        self.assertTrue(q["heterogeneous_hardware"])
        self.assertFalse(q["incomplete_provenance"])
        inv.add("unknown", {})
        self.assertTrue(inv.summary()["incomplete_provenance"])

    def test_delta_and_package_keep_inference_provenance(self):
        # Exercise wrappers with tiny maps and a stub packager, never an official submission.
        from repro import challenge
        from screening.official_foundation_io import write_dsp5, read_dsp5, disp_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = root / "predictions"
            out = root / "output"
            out.mkdir()
            row = {"scene": "s", "frame": "0001", "pair_id": "s/0001"}
            for cond in ("clean", "noise"):
                for eye in ("left", "right"):
                    meta = {
                        **row,
                        "condition": cond,
                        "reference_eye": eye,
                        "method": "Identity",
                        "checkpoint_sha256": "c",
                        "upstream_commit": "u",
                        "overlap": 0.7,
                        "precision": "fp16",
                        "precision_scope": (
                            "forward-only" if cond == "clean" else "whole-call"
                        ),
                        "GPU": "A",
                        "batch": 2,
                    }
                    write_dsp5(
                        disp_path(layout, {**row, "condition": cond}, eye),
                        np.full((2, 6), int(cond != "clean"), np.float32),
                        meta,
                        shape=(2, 6),
                    )
            observe = lambda cs: [{**row, "condition": c} for c in cs]

            def load(p, expected):
                return read_dsp5(p, expected, shape=(2, 6))

            with patch.object(challenge, "output", return_value=out), patch.object(
                challenge, "bind"
            ), patch.object(
                challenge, "observations", side_effect=observe
            ), patch.object(
                challenge, "read_dsp5", side_effect=load
            ):
                challenge.delta(
                    SimpleNamespace(
                        conditions=["noise"],
                        run_id="test",
                        clean=layout,
                        corrupt=layout,
                        limit=None,
                        clean_abs=0.474,
                    )
                )
            q = common.read(out / "summary.json")
            self.assertEqual(q["macro"][2]["delta_abs"], 1.0)
            self.assertTrue(q["inference_provenance"]["mixed_precision_scope"])
            tool = root / "packager"
            tool.write_text("stub")

            def run_stub(*a, **kw):
                (out / "disp1_robustness.hdf5").write_bytes(b"test-container")

            tools = {
                "executables": [
                    {"name": "disp1_robust_subsampling", "sha256": common.sha(tool)}
                ]
            }
            with patch.object(challenge, "output", return_value=out), patch.object(
                challenge, "bind"
            ), patch.object(
                challenge, "observations", side_effect=observe
            ), patch.object(
                challenge, "CONDITIONS", ["clean", "noise"]
            ), patch.object(
                challenge, "read", return_value=tools
            ), patch.object(
                challenge, "profile", return_value={}
            ), patch.object(
                challenge.subprocess, "run", side_effect=run_stub
            ):
                challenge.package(
                    SimpleNamespace(
                        kind="robust", tool=tool, layout=layout, run_id="package"
                    )
                )
            ready = common.read(out / "PACKAGE_READY.json")
            self.assertTrue(ready["inference_provenance"]["mixed_precision_scope"])
            self.assertFalse(ready["uploaded"])
