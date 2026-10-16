import unittest
import numpy as np
from repro.revision import consensus_scores, choose_consensus, subset_scores


class RevisionSelectors(unittest.TestCase):
    def test_subset_recomputes_rank_instead_of_truncating_full_rank(self):
        # Removing a candidate changes the ranks; compare with direct reranking.
        risk = np.array([[0, 3, 3], [1, 0, 4], [2, 1, 0]], float)
        ids = ["identity", "b", "c"]
        n = risk.shape[1]
        counts = {
            "candidate_ids": ids,
            "pixels": n,
            "less_counts": [[int((a < b).sum()) for b in risk] for a in risk],
            "equal_counts": [[int((a == b).sum()) for b in risk] for a in risk],
        }
        from scipy.stats import rankdata

        for selected in (ids, ["identity", "c"], ["identity"]):
            expected = rankdata(
                risk[[ids.index(k) for k in selected]], axis=0, method="average"
            ).mean(axis=1)
            got = subset_scores(counts, selected)
            np.testing.assert_array_equal(list(got.values()), expected)

    def test_consensus_selects_prediction_not_oracle_or_fusion(self):
        ids = ["identity", "joint_01", "joint_02"]
        d = np.array([[0, 100], [1, 2], [1, 3]], dtype=np.float32)
        scores = consensus_scores(d, ids)
        self.assertEqual(choose_consensus(scores, "p", "noise", "raft"), "joint_02")
        self.assertEqual(d.shape, (3, 2))

    def test_tie_is_reproducible_and_order_independent(self):
        a = {"identity": 0.0, "joint_01": 0.0}
        x = choose_consensus(a, "p", "clean", "raft")
        self.assertEqual(
            x, choose_consensus(dict(reversed(list(a.items()))), "p", "clean", "raft")
        )

    def test_invalid_disparity_is_not_masked(self):
        with self.assertRaises(ValueError):
            consensus_scores(np.array([[np.nan]]), ["identity"])


class SharedFixedBest(unittest.TestCase):
    @staticmethod
    def row(values, n=100, split="dev"):
        return {"pair_id": "p", "condition": "brightness", "split": split,
                "candidate_statistics": {
                    cid: {"full": {"n": n, "abs_sum": rate*n/100,
                                    "px1_count": rate*n/100, "d1_count": rate*n/100}}
                    for cid, rate in values.items()}}

    def test_common_compromise_not_individual_winners(self):
        from repro.revision_shared_fixed import rank_shared
        rows = {"raft": [self.row({"a": 1, "b": 9, "c": 4})],
                "acv": [self.row({"a": 9, "b": 1, "c": 4})]}
        self.assertEqual(rank_shared(rows)[0]["candidate_id"], "c")

    def test_model_weight_does_not_depend_on_valid_pixel_count(self):
        from repro.revision_shared_fixed import rank_shared
        rows = {"raft": [self.row({"a": 0, "b": 6}, 100)],
                "acv": [self.row({"a": 8, "b": 4}, 1000)]}
        self.assertEqual(rank_shared(rows)[0]["candidate_id"], "a")

    def test_heldout_row_rejected(self):
        from repro.revision_shared_fixed import rank_shared
        rows = {"raft": [self.row({"a": 1})],
                "acv": [self.row({"a": 2}, split="test")]}
        with self.assertRaises(ValueError):
            rank_shared(rows)


if __name__ == "__main__":
    unittest.main()
