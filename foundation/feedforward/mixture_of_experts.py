import torch
import torch.nn as nn
import torch.nn.functional as F
from foundation.feedforward.feedforward import Qwen23FeedForward

class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mode = cfg.get("MOE_mode", "deepseek").lower() # "deepseek" or "standard/normal"
        self.top_k = cfg.get("MOE_num_experts_per_token", 8)
        self.num_total_experts = cfg["MOE_num_total_experts"]
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"]

        self.deepseek_style = (self.mode == "deepseek")
        if self.deepseek_style:
            self.shared_expert = Qwen23FeedForward(cfg)

        self.gate = nn.Linear(self.emb_dim, self.num_total_experts, bias=False)
        self.routed_experts = nn.ModuleList([
            Qwen23FeedForward(self.emb_dim, self.hidden_dim)
            for _ in range(self.num_total_experts)
        ])

        self.aux_load_use_weight = cfg["MOE_aux_load_use_weight=True"]
        self.lb_loss_coef = cfg["MOE_lb_loss_coef"]
        self.ent_loss_coef = cfg["MOE_ent_loss_coef"]
        self.aux_loss_total = 0

        # self.bias_update_speed = 0.001
        # self.expert_bias = 0

    def forward(self, x):
        # *B, L, dim = x.shape
        B, L, dim = x.shape
        x_flat = x.view(-1, dim) #(B*L,dim)

        if not self.deepseek_style:
            logits = self.gate(x_flat) # (B * L, num_total_experts)
            topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1) #(B*L,k), (B*L,k)
            topk_weights = F.softmax(topk_scores, dim=-1) # (B * L, k)
        else:
            x_norm = F.normalize(x_flat, p=2, dim=-1) #(B*L,dim)
            gate_weight_norm = F.normalize(self.gate.weight, p=2, dim=-1) # (num_total_experts, emb_dim)
            logits = F.linear(x_norm, gate_weight_norm, self.gate.bias) # (B * L, num_total_experts)
            # logits_with_bias = logits + self.expert_bias

            scores = torch.sigmoid(logits) # (B * L, num_total_experts)
            topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1) #(B*L, k), (B*L, k)
            topk_weights = F.normalize(topk_scores, p=1, dim=-1) # (B*L, k) float
            # topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

            # if self.training:
            #     with torch.no_grad():
            #         # Calculate current load (fraction of tokens assigned to each expert)
            #         # One-hot encoding of selected experts
            #         mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)
            #         current_load = mask.mean(dim=0) # Shape: (num_experts,)
            #         # current_load = get_current_expert_load() # e.g., [0.2, 0.5, 0.3]
            #
            #         # Target load (uniform distribution)
            #         target_load = self.top_k / self.num_total_experts
            #
            #         # Update bias: If load > target, decrease bias. If load < target, increase bias.
            #         # bias = bias + speed * sign(target - current)
            #         error = target_load - current_load
            #         self.expert_bias += self.bias_update_speed * torch.sign(error)
            #
            #         # self.expert_bias.clamp_(-0.5, 0.5)

        final_routed_output = torch.zeros_like(x_flat) #(B*L, dim)
        for expert_idx in range(self.num_total_experts):
            mask = (topk_indices == expert_idx) #(B*L, k)
            if not mask.any():
                continue

            batch_indices, k_rank_indices = mask.nonzero(as_tuple=True) # (selected,) (selected,)
            expert_input = x_flat[batch_indices, ...] #(selected, dim)
            weights = topk_weights[..., batch_indices, k_rank_indices].unsqueeze(-1) #(selected, 1)

            expert_output = self.routed_experts[expert_idx](expert_input) #(selected, dim)
            final_routed_output.index_add_(0, batch_indices, expert_output * weights)

            """OLD
            token_mask = mask.any(dim=-1) (L,)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)  # selected_idx: (selected_tokens,)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)  # expert_input: (selected_tokens, dim)

            mask_selected = mask[selected_idx]  # mask_selected: (selected_tokens, k), k = top_k
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)  # slot_indices: (selected_tokens, 1)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx),  # topk_probs_flat: (N, k), after select: (selected_tokens, k)
                dim=-1,
                index=slot_indices  # slot_indices: (selected_tokens, 1)
            ).squeeze(-1)  # selected_probs: (selected_tokens,) """

        total_output = final_routed_output + self.shared_expert(x_flat) \
            if self.deepseek_style else final_routed_output

        # return total_output.contiguous().view(*x.shape[:-1], dim)

        scores = scores.view(B, L, self.num_total_experts) # (B, L, num_total_experts)
        probs_full = F.softmax(scores, dim=-1) # (B, L, num_total_experts)
        gating_probs_scattered = torch.zeros_like(scores) # (B, L, num_total_experts)
        gating_probs_scattered.scatter_(dim=-1, index=topk_indices, src=topk_weights) # (B, L, num_total_experts)

        # importance: the average "probability mass" per expert (from probs_full; sums to 1 if computed over experts)
        # load: the average "actual assignment share" per expert (sums to 1 if computed over experts)
        # lb_loss: num_experts * sum_i (importance[i] * load[i])
        # ent_loss: negative entropy of probs_full (entropy higher the better for balancing)

        importance = probs_full.mean(dim=(0, 1))
        # importance = probs_full.mean(dim=tuple(range(probs_full.dim() - 1))) # (num_total_experts,), sum=1

        # Choose either way to compute load
        if self.aux_load_use_weight:
            load = gating_probs_scattered.mean(dim=(0, 1))
            # load_raw = gating_probs_scattered.sum(dim=(0, 1)) # (num_total_experts,), sum=B*L
            # load = load_raw / (B * L) # (num_total_experts,), sum=1
        else:
            dispatch_one_hot = torch.zeros_like(scores) #(B, L, num_total_experts)
            dispatch_one_hot.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices)) #(B*L, num_total_experts)
            # load = dispatch_one_hot.sum(dim=(0, 1)) / (B * L * self.top_k)  # (num_total_experts,), sum=1
            load = dispatch_one_hot.mean(dim=(0, 1)) / self.top_k

        lb_loss = self.num_total_experts * torch.sum(importance * load)

        # ent_loss = E_{B,L}[ sum_e p_e * log(p_e) ]
        ent_loss = (probs_full * torch.log(probs_full.clamp_min(1e-6))).sum(dim=-1).mean()

        self.aux_loss_total = self.lb_loss_coef * lb_loss + self.ent_loss_coef * ent_loss
        aux: dict[str, torch.Tensor] = {
            "lb_loss": lb_loss,
            "ent_loss": ent_loss,
            "total": self.aux_loss_total,
        }
        return total_output.contiguous().view(*x.shape[:-1], dim) #, aux

def get_total_moe_aux_loss(model):
    total_aux_loss = 0.0
    for module in model.modules():
        if isinstance(module, MoEFeedForward):
            if hasattr(module, 'aux_loss_total'):
                total_aux_loss += module.aux_loss_total
    return total_aux_loss

# optimizer.zero_grad()
# output = model(input_ids)
# main_loss = criterion(output.view(-1, vocab_size), targets.view(-1))
# aux_loss = get_total_moe_aux_loss(model)
# total_loss = main_loss + aux_loss
