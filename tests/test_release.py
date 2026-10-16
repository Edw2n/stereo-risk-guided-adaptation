"""Release workflows: no inference, downloads or benchmark upload."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from repro import common, workflow
from repro.common import ROOT, read, write
from screening.official_foundation_io import disp_path, write_dsp5


class ReleaseWorkflows(unittest.TestCase):
    def test_profile_uses_only_requested_storage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "volume"
            profile = common.profile_for(root)
            for key, value in profile.items():
                for p in value if isinstance(value, list) else [value]:
                    self.assertTrue(Path(p).is_relative_to(root), key)

    def test_resume_rejects_changed_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(common, "implementation_sha256", return_value="version-one"):
                common.bind(root, {"task": "fixture"})
                common.bind(root, {"task": "fixture"})
            with patch.object(common, "implementation_sha256", return_value="version-two"):
                with self.assertRaises(ValueError):
                    common.bind(root, {"task": "fixture"})

    def test_native_index_requires_same_complete_cohort(self):
        ids = [c["id"] for c in read(ROOT / "configs/challenge/bank8.json")["candidates"]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for cid in ids:
                run = root / cid
                write(run / "RUN.json", {"spec": {"task": "challenge-infer", "preset": "croco-07-fp16",
                      "candidate": {"id": cid}, "condition": "clean", "pairs": ["0003/0001"]}})
                (run / "observations.jsonl").write_text(json.dumps({"pair_id": "0003/0001"}) + "\n")
                cache = run / "cache/clean/0003" / ("0001_" + cid + ".h5")
                cache.parent.mkdir(parents=True)
                cache.write_bytes(b"contents validated later by the selection reader")
                runs.append(run)
            target = root / "index"
            target.mkdir()
            args = SimpleNamespace(runs=runs, condition="clean", run_id="index")
            with patch.object(workflow, "output", return_value=target), patch.object(workflow, "bind"):
                workflow.index(args)
                self.assertEqual(set(read(target / "CACHE_INDEX.json")), set(ids))
                with self.assertRaises(ValueError):
                    workflow.index(SimpleNamespace(**{**vars(args), "runs": runs[:-1]}))
                (runs[0] / "observations.jsonl").write_text("")
                with self.assertRaises(ValueError):
                    workflow.index(args)

    def test_layout_rejects_method_mix_and_duplicate_condition(self):
        row = {"scene": "s", "frame": "0001", "pair_id": "s/0001", "condition": "clean"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            for eye in ("left", "right"):
                write_dsp5(disp_path(source, row, eye), np.ones((2, 6), np.float32),
                           {**row, "method": "Identity", "reference_eye": eye}, shape=(2, 6))
            target = root / "layout"
            target.mkdir()
            args = SimpleNamespace(sources=[source], method="Identity", run_id="layout")
            with patch.object(workflow, "CONDITIONS", ["clean"]), patch.object(
                workflow, "observations", return_value=[row]
            ), patch.object(workflow, "output", return_value=target), patch.object(workflow, "bind"):
                workflow.layout(args)
                self.assertEqual((target / "predictions/clean").resolve(), source / "clean")
                with self.assertRaises(ValueError):
                    workflow.layout(SimpleNamespace(**{**vars(args), "method": "tile_disparity_std_unweighted_global"}))
                with self.assertRaises(ValueError):
                    workflow.layout(SimpleNamespace(**{**vars(args), "sources": [source, source]}))

    def test_frozen_stage_config_remains_bound_to_portable_sources(self):
        from repro.common import digest, sha
        plan = read(ROOT / "configs/revision/camera_ready_20261012.json")
        fixed = read(ROOT / "configs/revision/shared_fixed_best_20261014.json")
        self.assertEqual(plan["base_config_sha256"], sha(ROOT / "configs/paper/adaptation.json"))
        self.assertEqual(plan["bank_sha256"], digest(read(ROOT / "configs/paper/bank.json")))
        self.assertEqual(fixed["plan_sha256"], digest(plan))


if __name__ == "__main__":
    unittest.main()
