import torch
import torch.nn as nn
import torch.nn.functional as F
from foundation.kullback_leibler_div import kl_full_distribution

def configure_student_model(student_model: nn.Module, unfreeze_last_n_blocks: int = 2):
    for p in student_model.parameters():
        p.requires_grad = False

    for block in student_model.transformer_blocks[-unfreeze_last_n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in student_model.final_norm.parameters():
        p.requires_grad = True

    for p in student_model.out_head.parameters():
        p.requires_grad = True

class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, ignore_index: int = -100):
        super().__init__()
        self.Temp = float(temperature)
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss(ignore_index=int(ignore_index))

    def forward(
        self,
        student_logits: torch.Tensor, # (B, L, V)
        teacher_logits: torch.Tensor, # (B, L, V)
        targets: torch.Tensor, # (B, L)
    ):
        B, L, V = student_logits.shape

        loss_ce = self.ce(
            student_logits.reshape(-1, V), # (B*L, V)
            targets.flatten(), # (B*L,)
        )

        logp_s = F.log_softmax(student_logits / self.Temp, dim=-1) # (B, L, V)
        logp_t = F.log_softmax(teacher_logits / self.Temp, dim=-1) # (B, L, V)

        loss_kd = kl_full_distribution(
            log_probs_policy=logp_t, # (B, L, V)
            log_probs_ref=logp_s, # (B, L, V)
        ).mean() * torch.pow(self.Temp, 2)

        total = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        return total, loss_ce, loss_kd


def train_step(student_model, teacher_model, dataloader, optimizer, device, config):
    criterion = DistillationLoss(
        temperature=config["temperature"],
        alpha=config["alpha"],
        ignore_index=config.get("ignore_index", -100),
    )

    student_model.train()
    teacher_model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch.to(device) # (B, L+1)
        inputs = input_ids[:, :-1] # (B, L)
        targets = input_ids[:, 1:] # (B, L)

        with torch.no_grad():
            teacher_logits = teacher_model(inputs) # (B, L, V)

        student_logits = student_model(inputs) # (B, L, V)

        loss, loss_ce, loss_kd = criterion(student_logits, teacher_logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Total={loss.item():.4f} (CE={loss_ce.item():.4f}, KD={loss_kd.item():.4f})")

    return total_loss / max(1, n_batches)
