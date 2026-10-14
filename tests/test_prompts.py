"""Forward/backward and public input-contract checks for the prompt extraction."""

import unittest

import torch

from kapt_lab.prompts import (
    GatedResidualPromptInjector,
    GraphMeanPromptGenerator,
    KAPTPromptModule,
    PromptFusion,
    TaskPromptPool,
)


class PromptTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.nodes = torch.randn(6, 3)
        self.batch = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)

    def test_variable_graph_sizes_and_full_backward(self):
        model = KAPTPromptModule(4, 3, 12, 8)
        nodes = self.nodes.clone().requires_grad_()
        prompted, graph_prompts = model(nodes, self.batch, torch.tensor([0, 1, 2]), 3)
        self.assertEqual(prompted.shape, (6, 3))
        self.assertEqual(graph_prompts.shape, (3, 8))
        self.assertFalse(torch.equal(prompted, nodes))
        (prompted.square().mean() + graph_prompts.square().mean()).backward()
        self.assertTrue(bool(torch.isfinite(nodes.grad).all()))
        self.assertGreater(nodes.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_selected_tasks_only_receive_prompt_gradients(self):
        pool = TaskPromptPool(4, 8)
        pool(torch.tensor([1, 1, 3]), 3).square().sum().backward()
        gradient = pool.prompts.grad
        self.assertEqual(gradient[0].abs().sum().item(), 0)
        self.assertEqual(gradient[2].abs().sum().item(), 0)
        self.assertGreater(gradient[1].abs().sum().item(), 0)
        self.assertGreater(gradient[3].abs().sum().item(), 0)

    def test_shared_task_matches_repeated_task_ids(self):
        pool = TaskPromptPool(4, 8)
        torch.testing.assert_close(pool(2, 3), pool(torch.tensor([2, 2, 2]), 3))

    def test_mean_pool_is_correct_and_permutation_invariant(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        encoded = generator.node_encoder(self.nodes)
        expected = torch.stack([encoded[self.batch == graph].mean(0) for graph in range(3)])
        result = generator(self.nodes, self.batch, 3)
        torch.testing.assert_close(result, expected)
        order = torch.tensor([5, 0, 4, 2, 3, 1])
        torch.testing.assert_close(generator(self.nodes[order], self.batch[order], 3), result)

    def test_node_injection_matches_defined_residual(self):
        injector = GatedResidualPromptInjector(3, 8)
        prompt = torch.randn(3, 8)
        projected = injector.proj(prompt[self.batch])
        expected = self.nodes + injector.gate(torch.cat((self.nodes, projected), -1)) * projected
        torch.testing.assert_close(injector(self.nodes, prompt, self.batch), expected)

    def test_double_dtype_can_be_used_explicitly(self):
        model = KAPTPromptModule(4, 3, 12, 8).double()
        prompted, graph_prompts = model(self.nodes.double(), self.batch, 1, 3)
        self.assertEqual(prompted.dtype, torch.float64)
        self.assertEqual(graph_prompts.dtype, torch.float64)

    def test_constructor_rejects_invalid_dimensions(self):
        for constructor in (
            lambda: TaskPromptPool(0, 8), lambda: TaskPromptPool(2, 7),
            lambda: TaskPromptPool(2, 8, num_tokens=0),
            lambda: TaskPromptPool(2, 8, num_heads=True),
            lambda: GraphMeanPromptGenerator(0, 8, 8),
            lambda: PromptFusion(8, -1), lambda: GatedResidualPromptInjector(3, 0),
        ):
            with self.subTest(constructor=constructor), self.assertRaises((ValueError, TypeError)):
                constructor()

    def test_task_id_contract(self):
        pool = TaskPromptPool(3, 8)
        for task in (-1, 3, True, 1.5, torch.tensor([0.0, 1.0]),
                     torch.tensor([0]), torch.tensor([[0, 1]]), torch.tensor([0, 3])):
            with self.subTest(task=task), self.assertRaises((ValueError, TypeError)):
                pool(task, 2)
        with self.assertRaises(ValueError):
            pool(0, 0)

    def test_node_shape_and_dtype_contract(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        invalid_nodes = (self.nodes[:, :2], self.nodes.unsqueeze(0), self.nodes.long(), self.nodes.double())
        for nodes in invalid_nodes:
            with self.subTest(shape=nodes.shape, dtype=nodes.dtype), self.assertRaises((ValueError, TypeError)):
                generator(nodes, self.batch, 3)

    def test_batch_shape_dtype_and_range_contract(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        invalid_batches = (self.batch.float(), self.batch[:-1], self.batch.unsqueeze(0),
                           torch.tensor([-1, 1, 1, 2, 2, 2]), torch.tensor([0, 1, 1, 3, 3, 3]))
        for batch in invalid_batches:
            with self.subTest(batch=batch), self.assertRaises((ValueError, TypeError)):
                generator(self.nodes, batch, 3)

    def test_empty_and_missing_graphs_are_rejected(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        with self.assertRaisesRegex(ValueError, "empty"):
            generator(torch.empty(0, 3), torch.empty(0, dtype=torch.long), 1)
        with self.assertRaisesRegex(ValueError, "at least one"):
            generator(self.nodes, self.batch, 4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            generator(self.nodes, torch.tensor([0, 0, 0, 2, 2, 2]), 3)

    def test_nonfinite_features_are_rejected(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        for invalid in (float("nan"), float("inf")):
            nodes = self.nodes.clone()
            nodes[0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "finite"):
                generator(nodes, self.batch, 3)

    def test_fusion_rejects_mismatched_or_empty_prompts(self):
        fusion = PromptFusion(8, 10)
        for task, graph in ((torch.randn(3, 8), torch.randn(2, 8)),
                            (torch.empty(0, 8), torch.empty(0, 8)),
                            (torch.randn(3, 8), torch.randn(3, 7))):
            with self.subTest(task_shape=task.shape, graph_shape=graph.shape), self.assertRaises(ValueError):
                fusion(task, graph)

    def test_injector_rejects_invalid_prompt_shape_and_dtype(self):
        injector = GatedResidualPromptInjector(3, 8)
        for prompt in (torch.randn(3, 7), torch.randn(3, 8).double(), torch.empty(0, 8)):
            with self.subTest(shape=prompt.shape, dtype=prompt.dtype), self.assertRaises((ValueError, TypeError)):
                injector(self.nodes, prompt, self.batch)

    def test_device_mismatch_is_rejected_before_computation(self):
        generator = GraphMeanPromptGenerator(3, 10, 8)
        with self.assertRaisesRegex(ValueError, "same device"):
            generator(torch.empty(6, 3, device="meta"), self.batch, 3)
        with self.assertRaisesRegex(ValueError, "same device"):
            generator(self.nodes, torch.empty(6, dtype=torch.long, device="meta"), 3)
        pool = TaskPromptPool(3, 8)
        with self.assertRaisesRegex(ValueError, "same device"):
            pool(torch.empty(3, dtype=torch.long, device="meta"), 3)


if __name__ == "__main__":
    unittest.main()
