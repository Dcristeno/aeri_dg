import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int):
        super().__init__()
        hidden_dim = max(dim // reduction, 1)
        self.down = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(x)))


class SharedMoEAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        reduction: int,
        residual_scale: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.residual_scale = residual_scale
        self.experts = nn.ModuleList(
            [BottleneckAdapter(dim=dim, reduction=reduction) for _ in range(num_experts)]
        )
        self.input_gate = nn.Linear(dim, num_experts, bias=False)
        self.domain_gate = nn.Linear(dim, num_experts, bias=False)
        self.gate_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens: torch.Tensor, domain_param: torch.Tensor) -> torch.Tensor:
        dtype = tokens.dtype
        tokens = tokens.float()
        domain_param = domain_param.float()

        input_logits = self.input_gate(tokens)
        domain_logits = self.domain_gate(domain_param).view(1, 1, -1)
        alpha = self.gate_alpha.sigmoid()
        gate_logits = (1.0 - alpha) * input_logits + alpha * domain_logits

        weights, selected = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1).to(tokens.dtype)

        mixed_delta = torch.zeros_like(tokens)
        for expert_idx, expert in enumerate(self.experts):
            batch_idx, token_idx, rank_idx = torch.where(selected == expert_idx)
            if batch_idx.numel() == 0:
                continue
            expert_delta = expert(tokens[batch_idx, token_idx])
            expert_weight = weights[batch_idx, token_idx, rank_idx].unsqueeze(-1)
            mixed_delta[batch_idx, token_idx] += expert_weight * expert_delta

        return (tokens + self.residual_scale * mixed_delta).to(dtype)


class AERIMoEAdapter(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        num_experts: int = 8,
        top_k: int = 4,
        reduction: int = 8,
        pool_topk: int = 4,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.pool_topk = max(1, pool_topk)
        self.shared_moe = SharedMoEAdapter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            reduction=reduction,
            residual_scale=residual_scale,
        )
        self.aerial_domain = nn.Parameter(torch.randn(dim))
        self.ground_domain = nn.Parameter(torch.randn(dim))
        self.text_domain = nn.Parameter(torch.randn(dim))
        self.aerial_proj = nn.Linear(dim, dim)
        self.ground_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.local_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.zeros_(self.aerial_proj.weight)
        nn.init.zeros_(self.aerial_proj.bias)
        nn.init.zeros_(self.ground_proj.weight)
        nn.init.zeros_(self.ground_proj.bias)
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def _topk_pool(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = tokens.float().norm(dim=-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        k = min(self.pool_topk, tokens.shape[1])
        topk_idx = scores.topk(k, dim=1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        selected_tokens = tokens.gather(1, gather_idx)

        if mask is None:
            return selected_tokens.mean(dim=1)

        selected_mask = mask.gather(1, topk_idx).unsqueeze(-1).to(selected_tokens.dtype)
        pooled = (selected_tokens * selected_mask).sum(dim=1)
        denom = selected_mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def _encode_visual(
        self,
        token_feats: torch.Tensor,
        domain_param: torch.Tensor,
        proj: nn.Module,
    ) -> torch.Tensor:
        cls_feat = token_feats[:, 0, :].float()
        patch_tokens = token_feats[:, 1:, :]
        adapted_tokens = self.shared_moe(patch_tokens, domain_param).float()
        local_feat = self._topk_pool(adapted_tokens)
        return F.normalize(cls_feat + self.local_scale * proj(local_feat), dim=-1)

    def _encode_text(self, text_feats: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        eos_pos = caption_ids.argmax(dim=-1)
        cls_idx = torch.arange(text_feats.shape[0], device=text_feats.device)
        eot_feat = text_feats[cls_idx, eos_pos].float()

        content_tokens = text_feats[:, 1:, :]
        token_pos = torch.arange(1, text_feats.shape[1], device=text_feats.device).view(1, -1)
        valid_mask = token_pos < eos_pos.view(-1, 1)
        adapted_tokens = self.shared_moe(content_tokens, self.text_domain).float()
        local_feat = self._topk_pool(adapted_tokens, valid_mask)
        return F.normalize(eot_feat + self.local_scale * self.text_proj(local_feat), dim=-1)

    def forward(
        self,
        aerial_tokens: torch.Tensor,
        ground_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        caption_ids: torch.Tensor,
    ):
        aerial_feat = self._encode_visual(aerial_tokens, self.aerial_domain, self.aerial_proj)
        ground_feat = self._encode_visual(ground_tokens, self.ground_domain, self.ground_proj)
        text_feat = self._encode_text(text_tokens, caption_ids)
        return aerial_feat, ground_feat, text_feat
