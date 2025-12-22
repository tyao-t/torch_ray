import tiktoken
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def text_to_token_ids(text, tokenizer, *, create_batch_dim=True):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded)
    if create_batch_dim:
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor # (1, num_tokens)

def token_ids_to_text(token_ids, tokenizer, *, remove_batch_dim=True):
    flat = token_ids.squeeze(0) if remove_batch_dim else token_ids  # (1, num_tokens) -> (num_tokens, )
    return tokenizer.decode(flat.tolist()) # (num_tokens, )

def text_to_token_ids(text, tokenizer, *, create_batch_dim=True):
    encoded = tokenizer.encode(text, add_special_tokens=True) 
    encoded_tensor = torch.tensor(encoded)
    if create_batch_dim:
        encoded_tensor = encoded_tensor.unsqueeze(0)
    return encoded_tensor

def one_hot(idx, vocab_size, d_out):
    embedding = torch.nn.Embedding(vocab_size, d_out)
    onehot = torch.nn.functional.one_hot(idx)
    torch.manual_seed(123)
    num_idx = max(idx)+1 # or it can be vocab_size
    linear = torch.nn.Linear(num_idx, d_out, bias=False)
    linear.weight = torch.nn.Parameter(embedding.weight.T)
    linear.weight
    linear(onehot.float()) == embedding(idx)
    
# if __name__ == "__main__":
#     with open("the-verdict.txt", "r", encoding="utf-8") as f:
#         raw_text = f.read()
#         dataloader = create_dataloader(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
#         data_iter = iter(dataloader)
#         first_batch = next(data_iter)
#         second_batch = next(data_iter)
#         print(second_batch)

#     vocab_size, output_dim = 6, 3
#     torch.manual_seed(123)
#     embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#     token_embeddings = token_embedding_layer(first_batch)
#     print(token_embeddings.shape)
#     # torch.Size([8, 4, 256])

#     context_length = 1024
#     pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
#     pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#     print(pos_embeddings.shape)
#     # torch.Size([4, 256])

#     input_embeddings = token_embeddings + pos_embeddings
#     print(input_embeddings.shape)
#     # torch.Size([8, 4, 256])