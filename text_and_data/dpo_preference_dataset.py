import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import AutoTokenizer
from text_and_data.text_processor import text_to_token_ids

# Sample: {'instruction': '...', 'input': '...', 'chosen': '...', 'rejected': '...'}

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text
class PreferenceDatasetDPO(Dataset):
    def __init__(self, data):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            prompt = format_input(entry)
            prompt_tokens = tokenizer.encode(prompt)
            
            chosen_full_text = f"{prompt}\n\n### Response:\n{entry['chosen']}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{entry['rejected']}"
            
            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": text_to_token_ids(chosen_full_text, tokenizer),
                "rejected": text_to_token_ids(rejected_full_text, tokenizer),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encoded_texts[index]

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    batch_data = {
        "prompt": [], "chosen": [], "rejected": [],
        "rejected_mask": [], "chosen_mask": []
    }

    max_length_common = 0
    for key in ["chosen", "rejected"]:
        current_max = max(len(item[key]) + 1 for item in batch)
        max_length_common = max(max_length_common, current_max)

    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            # Right padding Padding
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            mask[len(sequence):] = False
            
            if mask_prompt_tokens:
                mask[:len(item["prompt"])+2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        batch_data[key] = tensor_stack.to(device)

    # Output shape: {'chosen': (B, L), 'chosen_mask': (B, L), ...}
    return batch_data

custom_collate_fn_dpo = partial(
    custom_collate_fn,
    device="cuda",   
    mask_prompt_tokens=True,
    allowed_max_length=1024
)

train_data = {'instruction': '...', 'input': '...', 'chosen': '...', 'rejected': '...'}
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct", trust_remote_code=True)

train_loader = DataLoader(
    PreferenceDatasetDPO(train_data, tokenizer),
    batch_size=8,
    collate_fn=custom_collate_fn_dpo,
    shuffle=True
)

eos_token_id = tokenizer.eos_token_id
