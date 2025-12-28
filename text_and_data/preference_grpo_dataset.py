import torch
from torch.utils.data import Dataset, DataLoader

class GRPODataset(Dataset):
    def __init__(self, groups, tokenizer, max_length=None):
        self.groups = groups
        self.tok = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.eos_token_id  # simplified as requested
        # groups[i] = {
        #     "prompt": str,
        #     "responses": List[str]  # length = G
        # }

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # Each __getitem__ returns 1 GRPO group:
        # input_ids: (G, L) prompt+response, right-padded to max length within 1 grp
        # mask: (G, L) 1=response tokens, 0=prompt+pad

        # For simplicity, pad_token_id == eos_token_id == tokenizer.eos_token_id
        item = self.groups[idx]
        prompt_text = item["prompt"]
        responses = item["responses"]
        G = len(responses)

        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        seqs = []
        for r in responses:
            resp_ids = self.tok.encode(r, add_special_tokens=False)
            if len(resp_ids) == 0 or resp_ids[-1] != self.eos_token_id:
                resp_ids = resp_ids + [self.eos_token_id]

            full = prompt_ids + resp_ids
            if self.max_length is not None:
                full = full[: self.max_length]
            seqs.append(full)

        N = max(len(s) for s in seqs)
        input_ids = torch.full((G, N), self.pad_token_id, dtype=torch.long)
        completion_mask = torch.zeros((G, N), dtype=torch.bool)

        for i, s in enumerate(seqs):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)

            start = min(prompt_len, N)
            end = min(L, N)
            if end > start:
                completion_mask[i, start:end] = True

        return {
            "input_ids": input_ids,              # (G, N)
            "completion_mask": completion_mask,  # (G, N)
        }

def grpo_collate_fn(batch):
    # DataLoader(batch_size=1): return the single sample directly
    assert len(batch) == 1, "Use batch_size=1 so one batch == one GRPO group."
    return batch[0]

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)

    groups = [
        {"prompt": "Q: 1+1=?\nA:", "responses": ["2", "two", "It is 2."]},
        {"prompt": "Translate 'hello' to French.\nA:", "responses": ["bonjour", "salut", "bonjour!"]},
    ]

    ds = GRPODataset(groups, tokenizer, max_length=128)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=grpo_collate_fn)

    batch = next(iter(dl))
    print(batch["input_ids"].shape)        # (G, N)
    print(batch["completion_mask"].shape)  # (G, N)
