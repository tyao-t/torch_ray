import torch
from transformers import AutoTokenizer

from text_and_data.instruction_sft_dataset import *
def train_instruction_sfe_simple(model, train_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs.to(device)
            targets.to(device)
            # inputs shape: (B, L)
            # targets shape: (B, L)
            optimizer.zero_grad()
            
            # (B, L)
            logits = model(inputs)
            
            # CrossEntropy ignores targets with value -100
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), # (B*L, V)
                targets.flatten() # (B*L, )
            )
            
            loss.backward()
            optimizer.step()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)
train_data = {'instruction': '...', 'input': '...', 'output': '...'}
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    collate_fn=functools.partial(customized_collate_fn, tokenizer=tokenizer),
    shuffle=True,
    drop_last=True
)