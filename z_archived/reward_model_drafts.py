import torch
import torch.nn as nn
from foundation.operators.normalizations import RMSNorm
from foundation.transformer_blocks import Qwen3TransformerBlock

class Qwen3RewardModel(nn.Module):
    """
    Reward Model = Qwen3 backbone + scalar reward head.

    输入: input_ids [B, T], attention_mask [B, T] (可选, 1=有效token, 0=pad)
    输出: rewards [B] (每条序列一个标量分数)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.transformer_blocks = nn.ModuleList([Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])

        # 标量 reward head（最常见是线性层；bias 可有可无）
        self.reward_head = nn.Linear(cfg["emb_dim"], 1, bias=True, dtype=cfg["dtype"])

    @torch.no_grad()
    def _last_token_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        attention_mask: [B, T] with 1 for valid tokens.
        return: last valid token index per batch: [B]
        """
        # lengths = number of valid tokens
        lengths = attention_mask.long().sum(dim=1)  # [B]
        # clamp to avoid -1 if some row is all zeros (shouldn't happen in normal batches)
        idx = torch.clamp(lengths - 1, min=0)
        return idx

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        offset: int = 0,
        cache=None,
        mask: str = "causal",
        return_token_rewards: bool = False,
    ):
        """
        return_token_rewards=False:
            returns rewards [B]
        return_token_rewards=True:
            returns token_rewards [B, T], and sequence_rewards [B]
        """
        # x: [B, T, D]
        x = self.tok_emb(input_ids)

        for block in self.transformer_blocks:
            x = block(
                x, offset=offset, cache=cache, mask=mask,
                exact=self.cfg["exact"],
            )

        h = self.final_norm(x).to(self.cfg["dtype"])  # [B, T, D]

        # token-level reward (有些实现会只用最后 token；也有用每 token)
        token_rewards = self.reward_head(h).squeeze(-1)  # [B, T]

        # sequence-level pooling：取最后一个有效 token 的 reward
        if attention_mask is None:
            seq_rewards = token_rewards[:, -1]  # [B]
        else:
            last_idx = self._last_token_index(attention_mask)  # [B]
            seq_rewards = token_rewards.gather(1, last_idx.unsqueeze(1)).squeeze(1)  # [B]

        if return_token_rewards:
            return token_rewards, seq_rewards
        return seq_rewards

    @classmethod
    def from_base_model(cls, base_model: nn.Module, cfg: dict, *, init_reward_head: str = "zero"):
        """
        从已有 Qwen3Model 拷贝 backbone 参数，构造 RewardModel。
        init_reward_head:
          - "zero": reward_head 权重/偏置全 0（初始输出接近 0，训练更稳）
          - "normal": 默认初始化（nn.Linear 的默认）
        """
        rm = cls(cfg)
        # 拷贝 backbone
        rm.tok_emb.load_state_dict(base_model.tok_emb.state_dict())
        rm.transformer_blocks.load_state_dict(base_model.transformer_blocks.state_dict())
        rm.final_norm.load_state_dict(base_model.final_norm.state_dict())

        if init_reward_head == "zero":
            nn.init.zeros_(rm.reward_head.weight)
            if rm.reward_head.bias is not None:
                nn.init.zeros_(rm.reward_head.bias)

        return rm
class Qwen3ValueModel(nn.Module):
    """
    独立 Critic / Value 网络：
      输入: input_ids [B, T]
      输出:
        - token_values [B, T]  (每个 token 的 V(s_t)，PPO/GAE 常用)
        - 可选 seq_values [B]  (最后一个有效 token 的 value)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.transformer_blocks = nn.ModuleList([Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])

        # Critic head：输出标量 V(s_t)
        self.value_head = nn.Linear(cfg["emb_dim"], 1, bias=True, dtype=cfg["dtype"])

    @torch.no_grad()
    def _last_token_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        attention_mask: [B, T] (1=有效token, 0=pad)
        return: 每条序列最后一个有效 token 的 index, shape [B]
        """
        lengths = attention_mask.long().sum(dim=1)          # [B]
        idx = torch.clamp(lengths - 1, min=0)               # 避免出现 -1
        return idx

    def forward(
        self,
        in_token_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        offset: int = 0,
        cache=None,
        mask: str = "causal",
        return_seq_value: bool = False,
    ):
        # x: [B, T, D]
        x = self.tok_emb(in_token_ids)
        for block in self.transformer_blocks:
            x = block(
                x, offset=offset, cache=cache, mask=mask,
                exact=self.cfg["exact"],
            )

        h = self.final_norm(x).to(self.cfg["dtype"])        # [B, T, D]
        token_values = self.value_head(h).squeeze(-1)       # [B, T]

        if not return_seq_value:
            return token_values

        # seq value：取最后一个有效 token（或最后一个 token）
        if attention_mask is None:
            seq_values = token_values[:, -1]                # [B]
        else:
            last_idx = self._last_token_index(attention_mask)  # [B]
            seq_values = token_values.gather(1, last_idx.unsqueeze(1)).squeeze(1)

        return token_values, seq_values

    @classmethod
    def from_actor_backbone(cls, actor_model: nn.Module, cfg: dict, *, init_value_head: str = "zero"):
        """
        用 actor 的 backbone 初始化 critic（常见做法：让 critic 从同一个 SFT/actor 起步更稳）。
        只拷贝 tok_emb / blocks / final_norm；value_head 独立初始化。
        """
        vm = cls(cfg)
        vm.tok_emb.load_state_dict(actor_model.tok_emb.state_dict())
        vm.transformer_blocks.load_state_dict(actor_model.transformer_blocks.state_dict())
        vm.final_norm.load_state_dict(actor_model.final_norm.state_dict())

        if init_value_head == "zero":
            nn.init.zeros_(vm.value_head.weight)
            if vm.value_head.bias is not None:
                nn.init.zeros_(vm.value_head.bias)

        return vm

import torch
import torch.nn as nn
from foundation.operators.normalizations import RMSNorm
from foundation.transformer_blocks import Qwen3TransformerBlock

class Qwen3CriticModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 骨干网络 (Backbone) - 与 Actor 和 Reward Model 共享相同的结构
        # 这样你可以加载 Reward Model 的权重作为 Critic 的初始化
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.transformer_blocks = nn.ModuleList(
            [Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        
        # 2. Value Head
        # 结构上 = Reward Head。
        # 输入: Hidden State [Batch, Dim]
        # 输出: Value [Batch, 1] (代表当前状态的预期价值)
        self.value_head = nn.Linear(cfg["emb_dim"], 1, bias=False, dtype=cfg["dtype"])

    def forward(self, in_token_ids, *, offset=0, cache=None, mask="causal"):
        # --- Backbone 前向传播 (与 Actor/Reward 一致) ---
        x = self.tok_emb(in_token_ids)
        for block in self.transformer_blocks:
            x = block(
                x, offset=offset, cache=cache, mask=mask,
                exact=self.cfg["exact"],
            )
        
        x = self.final_norm(x).to(self.cfg["dtype"])
        
        # --- Value Head 前向传播 ---
        # x shape: [Batch, Seq_Len, Dim]
        # value shape: [Batch, Seq_Len, 1]
        values = self.value_head(x)
        
        # 压缩维度，输出 [Batch, Seq_Len]
        # PPO 训练时，我们需要每个 token 位置的 Value 来计算 Advantage
        values = values.squeeze(-1)
        
        return values
    
import torch
import torch.nn as nn
from foundation.operators.normalizations import RMSNorm
from foundation.transformer_blocks import Qwen3TransformerBlock

class Qwen3RewardModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 1. 骨干网络 (Backbone) - 与原模型完全一致，方便加载 SFT 权重
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.transformer_blocks = nn.ModuleList(
            [Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        
        # 2. 核心区别 (Reward Head)
        # 将输出维度从 vocab_size 改为 1
        # 注意：这里通常建议 bias=True，虽然你原模型 bias=False。
        # 因为 Reward 分数通常不是以 0 为中心的，bias 能帮助模型更快找到分数均值。
        self.reward_head = nn.Linear(cfg["emb_dim"], 1, bias=False, dtype=cfg["dtype"])

    def forward(self, in_token_ids, *, offset=0, cache=None, mask="causal"):
        # --- 前半部分与原模型完全一致 ---
        x = self.tok_emb(in_token_ids)
        for block in self.transformer_blocks:
            x = block(
                x, offset=offset, cache=cache, mask=mask,
                exact=self.cfg["exact"],
            )
        
        # 进行归一化
        x = self.final_norm(x).to(self.cfg["dtype"])
        
        # --- 后半部分：计算分数 ---
        # 现在的 x 形状是 [Batch, Seq_Len, Dim]
        # 经过 reward_head 后变成 [Batch, Seq_Len, 1]
        scores = self.reward_head(x)
        
        # 压缩维度：[Batch, Seq_Len, 1] -> [Batch, Seq_Len]
        # 这样每个 Token 对应一个分数
        scores = scores.squeeze(-1)
        
        return scores
