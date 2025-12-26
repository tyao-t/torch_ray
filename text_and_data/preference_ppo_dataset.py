import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import AutoTokenizer

class PPODataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=None):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_text = item["prompt"]
        response_text = item["response"]

        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        resp_ids = self.tok.encode(response_text, add_special_tokens=False)
        if len(resp_ids) == 0 or resp_ids[-1] != self.eos_token_id:
            resp_ids = resp_ids + [self.eos_token_id]

        full_ids = prompt_ids + resp_ids
        if self.max_length is not None:
            full_ids = full_ids[: self.max_length]

        return {
            "input_ids": full_ids,
            "prompt_len": prompt_len
        }

def ppo_collate_fn(batch, pad_token_id):
    B = len(batch)
    
    seqs = [item["input_ids"] for item in batch]
    prompt_lens = [item["prompt_len"] for item in batch]

    N = max(len(s) for s in seqs)
    input_ids = torch.full((B, N), pad_token_id, dtype=torch.long)
    completion_mask = torch.zeros((B, N), dtype=torch.bool)

    for i, s in enumerate(seqs):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long)

        p_len = prompt_lens[i]
        start = min(p_len, N)
        end = min(L, N)
        if end > start:
            completion_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
    }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = [
        {"prompt": "Q: 1+1=?\nA:", "response": "2"},
        {"prompt": "Q: 1+1=?\nA:", "response": "two"},
    ]

    ds = PPODataset(samples, tokenizer, max_length=128)
    
    collate_fn = partial(ppo_collate_fn, pad_token_id=tokenizer.pad_token_id)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(dl))
    print(batch["input_ids"].shape)
    print(batch["completion_mask"].shape)
