import torch
from torch.utils.data import Dataset, DataLoader
from text_and_data.text_processor import text_to_token_ids
import os
import tiktoken
from transformers import AutoTokenizer
import numpy as np
class PretrainDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len, stride, *, encode=True):
        self.input_ids, self.target_ids = [], []

        token_ids = text
        if encode: text_to_token_ids(token_ids, tokenizer, create_batch_dim=False)
        assert len(token_ids) > seq_len # > is due to target_chunk = token_ids[i+1:i+seq_len+1]

        for i in range(0, len(token_ids) - seq_len, stride):
            input_chunk = token_ids[i:i+seq_len]
            target_chunk = token_ids[i+1:i+seq_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class PretrainDatasetMmap(Dataset):
    def __init__(self, root_dir: str, split_name: str, seq_len: int, storage_dtype=np.uint16):
        self.seq_len = int(seq_len)
        self.storage_dtype = storage_dtype
        self.bin_path = os.path.join(root_dir, f"{split_name}.tokenids.bin")
        self.token_memmap = self._token_memmap = np.memmap(self.bin_path, dtype=self.storage_dtype, mode="r")
        self.total_num_tokens = len(self._token_memmap)

    def __len__(self) -> int:
        return max(0, self.total_num_tokens - self.seq_len)

    def __getitem__(self, idx: int):
        idx = int(idx)
        input_np = np.array(self.token_memmap[idx : idx + self.seq_len], dtype=np.int64)
        target_np = np.array(self.token_memmap[idx + 1 : idx + 1 + self.seq_len], dtype=np.int64)

        input_tokens = torch.from_numpy(input_np)
        target_tokens = torch.from_numpy(target_np)
        return input_tokens, target_tokens

def create_pretrain_dataloader(
    text, batch_size: int = 4, seq_len: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    *,
    encode: bool = True,
    tokenizer_name: str = "gpt2",
    device_type: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True) # tiktoken.get_encoding(tokenizer_name)
    dataset = PretrainDataset(text=text, tokenizer=tokenizer, seq_len=seq_len, stride=stride, encode=encode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
    )
    return loader

def create_pretrain_mmap_dataloader(
    text, *, batch_size: int = 4, seq_len: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    split_name = "train",
    root_dir = ".",
    device_type: str = "cuda",
):
    dataset = PretrainDatasetMmap(text=text, seq_len=seq_len, split_name=split_name, root_dir=root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
    )
    return loader
