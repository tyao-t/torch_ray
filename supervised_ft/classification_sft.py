import torch
from torch.utils.data import DataLoader
from text_and_data.classification_sft_dataset import SpamDataset, classification_collate_fn
from foundation.model import Qwen3Model
from transformers import AutoTokenizer

model = Qwen3Model(dict())
for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

for block in model.transformer_blocks[-2:]:
    for param in block.parameters():
        param.requires_grad = True

for param in [model.final_norm.parameters(), model.out_head.parameters()]:
    for p in param: p.requires_grad = True

def train_classifier_step(model, inputs, targets, last_indices, optimizer):
    optimizer.zero_grad()
    
    all_logits = model(inputs) # (B, Classes)
    
    # 不能直接用 [:, last_indices, :]，那会取回 (B, B, Classes)
    # 配合 torch.arange 来指定每一行的特定索引
    batch_size = inputs.shape[0]
    logits = all_logits[torch.arange(batch_size), last_indices, :] # (B, Classes)
    
    loss = torch.nn.functional.cross_entropy(logits, targets)
    
    loss.backward()
    optimizer.step()
    return loss.item()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)
train_loader = DataLoader(
    SpamDataset("train.csv", tokenizer, max_length=1024),
    batch_size=8,
    collate_fn=classification_collate_fn,
    shuffle=True
)