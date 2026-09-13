from __future__ import annotations

import unittest
from pathlib import Path

import torch

from graph_constraints import build_graph_tensors, gather_edge_lag_values, load_config
from validate_graph_constraints import validate


CONFIG = Path(__file__).resolve().parent / "config" / "graph_constraints_v2.json"


class GraphConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG)

    def test_contract(self) -> None:
        self.assertEqual(validate(self.config)["status"], "PASS")

    def test_optional_edges_are_additive(self) -> None:
        base = build_graph_tensors(self.config)
        optional = build_graph_tensors(self.config, include_optional_pcmci=True)
        self.assertEqual(
            optional["edge_index"].shape[1] - base["edge_index"].shape[1],
            len(self.config["optional_pcmci_edges"]),
        )

    def test_remove_self_memory(self) -> None:
        full = build_graph_tensors(self.config)
        cross = build_graph_tensors(self.config, include_self_memory=False)
        self.assertLess(cross["edge_index"].shape[1], full["edge_index"].shape[1])

    def test_lag_zero_and_one_read_t_and_t_minus_one(self) -> None:
        history = torch.arange(1 * 61 * 54, dtype=torch.float32).reshape(1, 61, 54, 1)
        edge_index = torch.tensor([[3], [7]], dtype=torch.long)
        mask = torch.zeros((1, 61), dtype=torch.bool)
        mask[0, 0:2] = True
        values, returned_mask = gather_edge_lag_values(history, edge_index, mask)
        self.assertEqual(values.shape, (1, 1, 2, 1))
        self.assertEqual(values[0, 0, 0, 0].item(), history[0, -1, 3, 0].item())
        self.assertEqual(values[0, 0, 1, 0].item(), history[0, -2, 3, 0].item())
        self.assertTrue(returned_mask[0, 0, 0, 0].item())
        self.assertTrue(returned_mask[0, 0, 1, 0].item())


if __name__ == "__main__":
    unittest.main()
