import torch
from torch.utils.data import Dataset, DataLoader
from text_and_data.text_processor import text_to_token_ids
import functools
# 1. DATA FORMATTING
# Initial Sample: {'instruction': '...', 'input': '...', 'output': '...'}

from transformers import AutoTokenizer
import torch

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# 2. DATASET DEFINITION
# Converts raw strings to token_ids (List[int]) during initialization
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            # Format: [ID_1, ID_2, ... ID_EOS]
            self.encoded_texts.append(text_to_token_ids(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# 3. COLLATE FUNCTION (TENSOR CONVERSION & MASKING)
# Transforms List[int] into padded Tensors [Batch_Size, Seq_Len]
def customized_collate_fn(
    batch, 
    ignore_index=-100, 
    pad_token_id=50256, 
    allowed_max_length=None, 
    device="cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] # Append EOS
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        # Inputs: All tokens except the last one [0, 1, 2]
        # Targets: All tokens shifted right by one [1, 2, 3]
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Masking: Replace all padding tokens except the first one with ignore_index (-100)
        mask = targets == pad_token_id
        mask[..., 0] = 0
        targets = targets.masked_fill_(mask=mask, value=ignore_index)

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
