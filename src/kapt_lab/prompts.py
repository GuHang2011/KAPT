"""Small, testable prompt components adapted from local ``kapt_components.py``.

The local project builds on KANO (https://github.com/ZJU-Fangyin/KANO),
distributed under MIT, Copyright (c) 2022 Fangyin. See LICENSE and
docs/source-and-changes.md for source attribution and the extraction boundary.
These components are a learning extraction, not a complete molecular model.
Graph prompts pool node features only; no edges or topology are consumed.
"""

from typing import Union

import torch
from torch import Tensor, nn


TaskIds = Union[int, Tensor]


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_features(features: Tensor, width: int, reference: Tensor, name: str) -> None:
    if not isinstance(features, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if features.ndim != 2 or features.shape[1] != width:
        raise ValueError(f"{name} must have shape [rows, {width}]")
    if not features.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if features.device != reference.device:
        raise ValueError(f"{name} and module parameters must use the same device")
    if features.dtype != reference.dtype:
        raise TypeError(f"{name} and module parameters must use the same dtype")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name} must contain finite values")


def _validate_batch(node_features: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
    _positive_int(num_graphs, "num_graphs")
    if not isinstance(batch_index, Tensor):
        raise TypeError("batch_index must be a torch.Tensor")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must have dtype torch.long")
    if batch_index.ndim != 1 or batch_index.shape[0] != node_features.shape[0]:
        raise ValueError("batch_index must have shape [num_nodes]")
    if batch_index.device != node_features.device:
        raise ValueError("batch_index and node_features must use the same device")
    if node_features.shape[0] == 0:
        raise ValueError("empty node batches are unsupported")
    if bool((batch_index < 0).any()) or bool((batch_index >= num_graphs).any()):
        raise ValueError("batch_index values must lie in [0, num_graphs)")
    counts = torch.bincount(batch_index, minlength=num_graphs)
    if bool((counts == 0).any()):
        raise ValueError("every graph must contain at least one node")
    return counts


class TaskPromptPool(nn.Module):
    """Learned task tokens, self-attention, then token mean pooling.

    ``task_id`` is one shared Python integer or a ``torch.long`` tensor of
    shape ``[num_graphs]`` on the same device as the parameters. The output
    has shape ``[num_graphs, prompt_dim]``. Only selected tasks receive gradients.
    """

    def __init__(self, num_tasks: int, prompt_dim: int, num_tokens: int = 5, num_heads: int = 4):
        super().__init__()
        for name, value in (("num_tasks", num_tasks), ("prompt_dim", prompt_dim),
                            ("num_tokens", num_tokens), ("num_heads", num_heads)):
            _positive_int(value, name)
        if prompt_dim % num_heads:
            raise ValueError("prompt_dim must be divisible by num_heads")
        self.num_tasks = num_tasks
        self.prompt_dim = prompt_dim
        self.prompts = nn.Parameter(torch.randn(num_tasks, num_tokens, prompt_dim) * 0.02)
        self.attention = nn.MultiheadAttention(prompt_dim, num_heads, batch_first=True)

    def forward(self, task_id: TaskIds, num_graphs: int) -> Tensor:
        _positive_int(num_graphs, "num_graphs")
        if isinstance(task_id, bool):
            raise TypeError("task_id must be an integer or a torch.long tensor")
        if isinstance(task_id, int):
            if not 0 <= task_id < self.num_tasks:
                raise ValueError("task_id is outside the prompt pool")
            tokens = self.prompts[task_id].unsqueeze(0).expand(num_graphs, -1, -1)
        elif isinstance(task_id, Tensor):
            if task_id.dtype != torch.long:
                raise TypeError("task_id tensor must have dtype torch.long")
            if task_id.ndim != 1 or task_id.shape[0] != num_graphs:
                raise ValueError("task_id tensor must have shape [num_graphs]")
            if task_id.device != self.prompts.device:
                raise ValueError("task_id and module parameters must use the same device")
            if bool((task_id < 0).any()) or bool((task_id >= self.num_tasks).any()):
                raise ValueError("task_id is outside the prompt pool")
            tokens = self.prompts[task_id]
        else:
            raise TypeError("task_id must be an integer or a torch.long tensor")
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        return attended.mean(dim=1)


class GraphMeanPromptGenerator(nn.Module):
    """Encode each node with an MLP, then mean-pool each graph.

    Inputs: float node features ``[num_nodes, node_dim]`` and long graph IDs
    ``[num_nodes]``. IDs may be unsorted but must cover ``0..num_graphs-1``.
    Empty graphs are rejected. Output: ``[num_graphs, prompt_dim]``.
    This is feature pooling, not an edge-aware graph encoder.
    """

    def __init__(self, node_dim: int, hidden_dim: int, prompt_dim: int):
        super().__init__()
        for name, value in (("node_dim", node_dim), ("hidden_dim", hidden_dim),
                            ("prompt_dim", prompt_dim)):
            _positive_int(value, name)
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, node_features: Tensor, batch_index: Tensor, num_graphs: int) -> Tensor:
        _validate_features(node_features, self.node_dim, self.node_encoder[0].weight, "node_features")
        counts = _validate_batch(node_features, batch_index, num_graphs)
        encoded = self.node_encoder(node_features)
        sums = encoded.new_zeros((num_graphs, self.prompt_dim))
        sums.index_add_(0, batch_index, encoded)
        return sums / counts.to(dtype=encoded.dtype).unsqueeze(1)


class PromptFusion(nn.Module):
    """Fuse matching task and graph prompts with a small MLP."""

    def __init__(self, prompt_dim: int, hidden_dim: int):
        super().__init__()
        _positive_int(prompt_dim, "prompt_dim")
        _positive_int(hidden_dim, "hidden_dim")
        self.prompt_dim = prompt_dim
        self.fusion = nn.Sequential(
            nn.Linear(prompt_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, task_prompt: Tensor, graph_prompt: Tensor) -> Tensor:
        reference = self.fusion[0].weight
        _validate_features(task_prompt, self.prompt_dim, reference, "task_prompt")
        _validate_features(graph_prompt, self.prompt_dim, reference, "graph_prompt")
        if task_prompt.shape != graph_prompt.shape or task_prompt.shape[0] == 0:
            raise ValueError("task_prompt and graph_prompt must have equal, nonempty shapes")
        return self.fusion(torch.cat((task_prompt, graph_prompt), dim=-1))


class GatedResidualPromptInjector(nn.Module):
    """Broadcast graph prompts to nodes and apply ``x + sigmoid(gate) * projected``.

    Inputs: nodes ``[num_nodes, node_dim]``, graph prompts
    ``[num_graphs, prompt_dim]``, and long graph IDs ``[num_nodes]``.
    All floating tensors must match parameter dtype/device. Output preserves
    node shape. Empty graphs and batches are rejected.
    """

    def __init__(self, node_dim: int, prompt_dim: int):
        super().__init__()
        _positive_int(node_dim, "node_dim")
        _positive_int(prompt_dim, "prompt_dim")
        self.node_dim = node_dim
        self.prompt_dim = prompt_dim
        self.proj = nn.Linear(prompt_dim, node_dim)
        self.gate = nn.Sequential(nn.Linear(node_dim * 2, 1), nn.Sigmoid())

    def forward(self, node_features: Tensor, graph_prompt: Tensor, batch_index: Tensor) -> Tensor:
        _validate_features(node_features, self.node_dim, self.proj.weight, "node_features")
        _validate_features(graph_prompt, self.prompt_dim, self.proj.weight, "graph_prompt")
        _validate_batch(node_features, batch_index, graph_prompt.shape[0])
        projected = self.proj(graph_prompt[batch_index])
        gate = self.gate(torch.cat((node_features, projected), dim=-1))
        return node_features + gate * projected


class KAPTPromptModule(nn.Module):
    """Compose task tokens, feature pooling, fusion, and actual node injection.

    ``forward(nodes, batch_index, task_id, num_graphs)`` returns a tuple:
    prompted nodes ``[num_nodes, node_dim]`` and fused graph prompts
    ``[num_graphs, prompt_dim]``. This small module needs an external encoder,
    prediction head, dataset, and training protocol for molecular prediction.
    """

    def __init__(self, num_tasks: int, node_dim: int, hidden_dim: int, prompt_dim: int,
                 num_tokens: int = 5, num_heads: int = 4):
        super().__init__()
        self.task_pool = TaskPromptPool(num_tasks, prompt_dim, num_tokens, num_heads)
        self.graph_pool = GraphMeanPromptGenerator(node_dim, hidden_dim, prompt_dim)
        self.fusion = PromptFusion(prompt_dim, hidden_dim)
        self.injector = GatedResidualPromptInjector(node_dim, prompt_dim)

    def forward(self, node_features: Tensor, batch_index: Tensor, task_id: TaskIds,
                num_graphs: int) -> tuple[Tensor, Tensor]:
        graph_prompt = self.graph_pool(node_features, batch_index, num_graphs)
        task_prompt = self.task_pool(task_id, num_graphs)
        fused_prompt = self.fusion(task_prompt, graph_prompt)
        return self.injector(node_features, fused_prompt, batch_index), fused_prompt
