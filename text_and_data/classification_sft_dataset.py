import torch
from torch.utils.data import Dataset
import pandas as pd

# Sample: {'Text': '...', 'Label': 0/1}
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length:
            self.encoded_texts = [t[:max_length] for t in self.encoded_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encoded_texts[index], self.data.iloc[index]["Label"]

def classification_collate_fn(batch, pad_token_id=50256, device="cuda"):
    batch_max_length = max(len(item[0]) for item in batch)
    inputs_lst, targets_lst, last_indices_lst = [], [], []

    for item_tokens, item_label in batch:
        last_indices_lst.append(len(item_tokens) - 1)
        
        padded = item_tokens + [pad_token_id] * (batch_max_length - len(item_tokens))
        inputs_lst.append(torch.tensor(padded))
        targets_lst.append(torch.tensor(item_label))

    inputs = torch.stack(inputs_lst).to(device) # (B, Max_L)
    targets = torch.stack(targets_lst).to(device) # (B,)
    last_indices = torch.tensor(last_indices_lst).to(device) # (B,)

    return inputs, targets, last_indices
