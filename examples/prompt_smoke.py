"""Run a synthetic prompt forward/backward pass; no dataset or trained weights."""

import torch

from kapt_lab.prompts import KAPTPromptModule


def main():
    torch.manual_seed(17)
    nodes = torch.randn(6, 3, requires_grad=True)
    batch_index = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)
    task_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    module = KAPTPromptModule(num_tasks=4, node_dim=3, hidden_dim=12, prompt_dim=8)
    prompted_nodes, graph_prompts = module(nodes, batch_index, task_ids, num_graphs=3)
    loss = prompted_nodes.square().mean() + graph_prompts.square().mean()
    loss.backward()
    assert nodes.grad is not None and bool(torch.isfinite(nodes.grad).all())
    print(f"PyTorch: {torch.__version__}")
    print(f"Node output: {tuple(prompted_nodes.shape)}")
    print(f"Graph prompt: {tuple(graph_prompts.shape)}")
    print("Synthetic forward/backward: passed")
    print("This smoke check does not evaluate molecular prediction performance.")


if __name__ == "__main__":
    main()
