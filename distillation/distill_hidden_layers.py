import torch
import torch.nn as nn
import torch.nn.functional as F

from foundation.operators.normalizations import RMSNorm
from foundation.transformer_blocks import Qwen3TransformerBlock
from distillation.distill_logits import DistillationLoss

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Tempok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.Tempransformer_blocks = nn.ModuleList([Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_token_ids, *, offset=0, cache=None, mask="causal"):
        x = self.Tempok_emb(in_token_ids) # (B, L, D)
        hiddens = [] # list[(B, L, D)]
        for block in self.Tempransformer_blocks:
            x = block(x, offset=offset, cache=cache, mask=mask, exact=self.cfg["exact"]) # (B, L, D)
            hiddens.append(x) # (B, L, D)
        x = self.final_norm(x).to(self.cfg["dtype"]) # (B, L, D)
        logits = self.out_head(x) # (B, L, V)
        return logits, hiddens # logits: (B, L, V), hiddens: list[(B, L, D)]

class MultiLayerAdaptationNetwork(nn.Module):
    def __init__(self, student_dim, teacher_dim, student_layers, teacher_layers, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.layer_mapping = {
            i: round(i * (teacher_layers - 1) / (student_layers - 1))
            for i in range(student_layers)
        }
        self.projections = nn.ModuleList(
            [nn.Linear(student_dim, teacher_dim, bias=True, dtype=dtype) for _ in range(student_layers)]
        )

    def forward(self, student_hidden_states):
        return [self.projections[i](h.to(self.dtype)) for i, h in enumerate(student_hidden_states)] # list[(B, L, D_t)]

class HiddenStateDistillationLoss(nn.Module):
    def __init__(self, adaptor, temperature=2.0):
        super().__init__()
        self.adaptor = adaptor
        self.Temp = float(temperature)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_hiddens, teacher_hiddens):
        adapted = self.adaptor(student_hiddens) # list[(B, L, D_t)]
        total_hidden_loss = 0.0

        for s_idx, t_idx in self.adaptor.layer_mapping.items():
            s_h = adapted[s_idx] # (B, L, D_t)
            t_h = teacher_hiddens[t_idx].detach() # (B, L, D_t)
            s_logprob = F.log_softmax(s_h / self.Temp, dim=-1) # (B, L, D_t)
            t_prob = F.softmax(t_h / self.Temp, dim=-1) # (B, L, D_t)
            total_hidden_loss = total_hidden_loss + self.kl_loss(s_logprob, t_prob) * torch.pow(self.Temp, 2)

        # teacher_dim = teacher_hiddens[0].size(-1)
        avg_hidden_loss = (total_hidden_loss / len(self.adaptor.layer_mapping)) # / teacher_dim

        return avg_hidden_loss

def train_step(cfg, student, teacher, adaptor, dataloader, optimizer, device):
    criterion1 = HiddenStateDistillationLoss(
        adaptor,
        temperature=cfg["temperature"]
    )

    criterion2 = DistillationLoss(
        temperature=cfg.get("logits_temperature", cfg["temperature"]),
        alpha=cfg.get("logits_alpha", 0.5),
        ignore_index=cfg.get("ignore_index", -100),
    )

    weight_hidden = float(cfg.get("weight_hidden_total", 1.0))
    weight_logits = float(cfg.get("weight_logits_total", 1.0))

    student.train()
    adaptor.train()
    teacher.eval()

    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device) # (B, L_total)
        inputs = input_ids[:, :-1] # (B, L)
        targets = input_ids[:, 1:] # (B, L)

        with torch.no_grad():
            teacher_logits, teacher_hiddens = teacher(inputs) # teacher_logits: (B, L, V)

        student_logits, student_hiddens = student(inputs) # student_logits: (B, L, V)

        hid_loss = criterion1(student_hiddens, teacher_hiddens)
        loss2, ce2, kd = criterion2(student_logits, teacher_logits, targets)

        loss = weight_hidden * hid_loss + weight_logits * loss2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(adaptor.parameters()), max_norm=1.0)
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx}: Total={loss.item():.4f} | "
                f"Hidden={hid_loss.item():.4f} | "
                f"LogitsTotal={loss2.item():.4f} (CE={ce2.item():.4f}, KD={kd.item():.4f})"
            )
            
# https://gemini.google.com/share/bbbc17ba5aee

# KL(P_ref‖Q_model)（正向 KL）：

# 对 Q 在 P>0 处给 0 概率会有“无限大”惩罚 → 鼓励 覆盖所有老师/参考的模式（mode covering）。

# KL(Q_model‖P_ref)（反向 KL）：

# 对在 P≈0 处给出概率会被重罚 → 鼓励 只贴某个高峰（mode seeking），容易模式塌缩；在蒸馏里通常不是默认选择，但在某些 RL/模仿学习里会用到。