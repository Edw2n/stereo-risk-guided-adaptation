import unittest, json, hashlib, tempfile
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from repro.common import ROOT, read
from sensor_v4_metrics import native_maps, sufficient, native_values, risk_points
from generic_transform_v1 import transform
from table2_pool_v4 import pool_coordinates
from screening.croco_proxy_metrics import rank_select
from screening.croco_tier8_evaluation import select_arrays, RAW
from screening.official_foundation_io import (
    reference_inputs,
    native_output,
    write_dsp5,
    read_dsp5,
)
from screening.official_foundation_delta_metrics import disagreement


class ProtocolTests(unittest.TestCase):
    def test_native_gt_units_and_d1(self):
        pred = np.array([[4.0, 4.0, 104.0]], np.float32)
        gt = np.array([[0.0, 100.0, 100.0]], np.float16)
        e, d, v, c = native_maps(pred, gt)
        np.testing.assert_array_equal(
            d, [[True, True, False]]
        )  # >3 alone is insufficient
        gt2 = np.repeat(np.repeat(gt, 2, 0), 2, 1)
        gt2[0, 0] = 4
        e2, _, _, _ = native_maps(pred, gt2)
        self.assertEqual(e2[0, 0], 0)
        self.assertEqual(e2[0, 2], 4)
        self.assertTrue(v.all())
        self.assertEqual(native_values(sufficient(e, d, v))["D1"], 200 / 3)

    def test_invalid_prediction_rejected(self):
        with self.assertRaises(ValueError):
            native_maps(np.array([[np.nan]], np.float32), np.ones((1, 1)))

    def test_gt_range_is_secondary(self):
        e, d, v, c = native_maps(
            np.array([[193.0]], np.float32), np.array([[200.0]], np.float32)
        )
        self.assertTrue(v.item())
        self.assertFalse(c.item())
        self.assertEqual(sufficient(e, d, v)["n"], 1)

    def test_sample_pool_nested_shared(self):
        valid = np.ones((40, 80), bool)
        a = pool_coordinates(valid, "0001/0005", 100, 20260925)
        b = pool_coordinates(valid, "0001/0005", 200, 20260925)
        np.testing.assert_array_equal(a, b[:100])
        self.assertTrue(np.all(a % 80 >= 4))

    def test_rank_not_raw_mean(self):
        # Candidate 0 wins two out of three pixels despite an extreme third value.
        risk = np.array(
            [[[0, 0, 0, 0, 0, 0, 100]], [[0, 0, 0, 0, 1, 1, 0]]], np.float32
        )
        disp = np.ones_like(risk) * np.arange(2, dtype=np.float32)[:, None, None]
        r = rank_select(risk, disp, "p", "noise", "s", tie_seed=9137)
        self.assertEqual(r["global_index"], 0)
        np.testing.assert_array_equal(r["global_disparity"][:, :4], disp[0, :, :4])

    def test_selector_both_eyes_and_ties(self):
        d = np.stack([np.zeros((2, 2, 9), np.float32), np.ones((2, 2, 9), np.float32)])
        raw = {
            k: np.stack(
                [np.ones((2, 2, 9), np.float32), np.zeros((2, 2, 9), np.float32)]
            )
            for k in RAW
        }
        a, info = select_arrays(d, raw, "s/f", "clean", ["identity", "joint_03"])
        b, _ = select_arrays(d, raw, "s/f", "clean", ["identity", "joint_03"])
        np.testing.assert_array_equal(
            a["tile_disparity_std_unweighted/global"],
            b["tile_disparity_std_unweighted/global"],
        )
        self.assertTrue(
            (a["tile_disparity_std_unweighted/global"][0, :, :4] == 0).all()
        )
        self.assertTrue(
            (a["tile_disparity_std_unweighted/global"][1, :, -4:] == 0).all()
        )
        self.assertTrue(
            (a["tile_disparity_std_unweighted/global"][0, :, 4:] == 1).all()
        )

    def test_transform_frozen_reference(self):
        fixtures = read(ROOT / "tests/transform_reference.json")
        rgb = np.random.default_rng(123).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        for c in read(ROOT / "configs/challenge/bank8.json")["candidates"]:
            self.assertEqual(
                hashlib.sha256(transform(rgb, c).tobytes()).hexdigest(),
                fixtures[c["id"]],
            )

    def test_risk_auc_orientation(self):
        y = np.array([False, False, True, True])
        risk = np.arange(4, dtype=np.float32)
        self.assertEqual(risk_points(y, risk, risk)["AUROC"], 1)
        self.assertEqual(risk_points(y, -risk, risk)["AUROC"], 0)

    def test_delta_is_prediction_difference(self):
        a = np.ones((2, 3), np.float32)
        b = a * 4
        self.assertEqual(disagreement(a, b)["delta_abs"], 3)

    def test_official_io_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.dsp5"
            d = np.arange(12, dtype=np.float32).reshape(3, 4)
            write_dsp5(p, d, {"reference_eye": "right"}, shape=d.shape)
            got, _ = read_dsp5(p, {"reference_eye": "right"}, shape=d.shape)
            np.testing.assert_array_equal(d, got)
            np.testing.assert_array_equal(
                native_output(native_output(d, "right"), "right"), d
            )

    def test_split_assignment(self):
        rows = read(ROOT / "configs/paper/split500.json")["records"]
        dev = {r["scene"] for r in rows if r["split"] == "dev"}
        test = {r["scene"] for r in rows if r["split"] == "test"}
        self.assertEqual((len(dev), len(test)), (25, 12))
        self.assertFalse(dev & test)
        self.assertEqual(sum(r["split"] == "dev" for r in rows), 328)
        for seed in [101, 202, 303]:
            a = read(ROOT / "configs/paper/assignments" / f"seed{seed}.json")["records"]
            self.assertEqual(len(a), 172)
            self.assertTrue(all(r["scene"] in test for r in a))

    def test_three_seed_published_report(self):
        from repro.paper import summarize_adaptation

        actual = summarize_adaptation(ROOT / "evidence/paper")
        for rows, name in zip(
            actual,
            ["three_seed_metrics", "three_seed_D1", "per_seed_metrics", "per_seed_D1"],
        ):
            self.assertEqual(
                rows,
                read(ROOT / "evidence/paper/reference_three_seed" / (name + ".json")),
            )


class BoundaryTests(unittest.TestCase):
    def test_native_float16_relative_threshold(self):
        gt = np.array([[80.0625]], np.float16)
        pred = gt.astype(np.float32) - np.float32(4.0035)
        error, d1, valid, _ = native_maps(pred, gt)
        self.assertGreater(error.item(), 0.05 * float(gt.item()))
        self.assertFalse(d1.item())  # upstream threshold multiplication retains float16

    def test_final_paired_batch_padding(self):
        from screening.croco_native import pack_reference_inputs

        left = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        right = left + 30
        a, b, pad = pack_reference_inputs([({}, left, right, left, right, 0)], 4)
        self.assertEqual(pad, 2)
        np.testing.assert_array_equal(a[0], left)
        np.testing.assert_array_equal(a[1], right[:, ::-1])
        np.testing.assert_array_equal(a[2], a[0])
        np.testing.assert_array_equal(a[3], a[1])

    def test_dsp5_pair_metadata_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.dsp5"
            d = np.ones((2, 3), np.float32)
            write_dsp5(
                path, d, {"pair_id": "a", "reference_eye": "left"}, shape=d.shape
            )
            with self.assertRaises(ValueError):
                read_dsp5(path, {"pair_id": "b"}, shape=d.shape)
