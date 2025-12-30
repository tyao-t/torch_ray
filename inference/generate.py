import torch
from inference.kv_cache import FullKvCache
from text_and_data.text_processor import text_to_token_ids, token_ids_to_text
import torch
from functools import partial
from inference.sampler import greedy_sampler
# logits = logits[:, -1, :]
# logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
# logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

@torch.no_grad() # torch.inference_mode()
def generate_text_stream_concat_flex(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=True, 
    generate_func=None,
    **generate_kwargs
):
    if generate_func is None:
        generate_func = generate_with_kv_cache
        
    input_ids = text_to_token_ids(prompt).to(device)
 
    generated_ids = []
    for token in generate_func(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        **generate_kwargs,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())
 
        if verbose:
            print(
                token_ids_to_text(next_token_id),
                end="",
                flush=True
            )
    return token_ids_to_text(generated_ids) #.replace("\n", " ") 

@torch.no_grad() # torch.inference_mode()
def generate_text_simple(model, token_ids, max_new_tokens, context_size, sampler=greedy_sampler):
    # token_ids: (batch_size, seq_len)
    idx = token_ids  # (batch, seq_len)
    model.eval()
    for _ in range(max_new_tokens):
        token_ids = token_ids[:, -context_size:]  # (batch, min(seq_len, context_size))
        logits = model(token_ids)  # (batch, seq_len, vocab_size)
        logits = logits[:, -1:, :]  # (batch, vocab_size)
        idx_next = sampler(logits)
        yield idx_next # (batch_size, 1)
        # idx_next = torch.argmax(logits, dim=-1, keepdim=False)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len+1)
        # seq_len increments by 1 each iteration
    return idx  # (batch, original_seq_len + max_new_tokens, 1)

@torch.no_grad()
def step(model, tok_indices, offset, kv_cache, *, mask="causal", n_tokens=1, sampler=greedy_sampler):
    if tok_indices.dim() == 1:
        tok_indices = tok_indices.unsqueeze(0)
    # with torch.no_grad():
    logits = model(tok_indices, offset=offset, kv_cache=kv_cache, mask=mask)
    logits = logits[:, -n_tokens:, :]
    # logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    output = sampler(logits)  # (batch, num_new_tokens)
    return output #, logprobs

@torch.no_grad()
def prefill(model, tok_indices, kv_cache, *, sampler=greedy_sampler):
    model.eval()
    token = step(model, tok_indices, offset=0, kv_cache=kv_cache, sampler=sampler)
    offset = tok_indices.numel()
    return token, offset

@torch.no_grad() # torch.inference_mode()
def generate_with_kv_cache(model, tok_indices, max_new_tokens, sampler=greedy_sampler):
    kv_cache = [FullKvCache() for _ in range(len(model.transformer_blocks))]
    model.eval()
    new_token, offset = prefill(model, tok_indices, kv_cache, sampler=sampler)
    yield new_token
    for _ in range(max_new_tokens):
        new_token = step(model, new_token, offset, kv_cache, sampler=sampler)
        yield new_token
        # if new_token.item() == eos_token_id:
        #     break
        offset += new_token.numel()
        # print(token_ids_to_text(new_token, tokenizer=None))
        # tok_indices = torch.cat((tok_indices, token), dim=1)
        # return tok_indices

@torch.no_grad()
def revert_cache(kv_cache, revert_len):
    for layer in kv_cache: layer.revert(revert_len)

# Below is Speculative decoding
@torch.no_grad() # torch.inference_mode()
def speculative_generate(draft_model, model, tok_indices, *, num_drafts=5):
    draft_kv_cache = [FullKvCache() for _ in range(len(draft_model.transformer_blocks))]
    kv_cache = [FullKvCache() for _ in range(len(model.transformer_blocks))]
    # Need to assume same tokenizer for both models
    draft_model.eval()
    draft_token, draft_offset = prefill(
        draft_model, tok_indices, draft_kv_cache
    )
    model.eval()
    token, offset = prefill(model, tok_indices, kv_cache)
    # assert draft_token.shape. == token.shape

    def draft_generate(model, last_token, offset, kv_cache, num_drafts):
        ret_tokens = torch.empty(1, 0, dtype=last_token.dtype) # (batch_size, num_tokens)
        for _ in range(num_drafts):
            new_token = step(model, last_token, offset, kv_cache)
            # ret_tokens = new_token if ret_tokens is None else torch.concat([ret_tokens, new_token], dim=-1)
            ret_tokens = torch.concat([ret_tokens, new_token], dim=-1)
            last_token = new_token
            offset += 1
        return ret_tokens

    all_correct, advance_new_token = True, None
    while True:
        # Below is an extra optimization (by tianhao.yao) but commented out for readability 
        # if all_correct and advance_new_token is not None:
        #     input_tokens = torch.concat(draft_generated_tokens[...,-1], advance_new_token, dim=-1)
        #     tokens = step(draft_model, input_tokens, draft_offset, kv_cache, n_tokens=2)
        #     draft_offset += 2
        #     draft_generated_tokens = torch.concat(tokens[...,-1], draft_generate(
        #         draft_model, token, draft_offset, draft_kv_cache, num_drafts-1
        #     ), dim=-1) # draft_generated_tokens.shape: (1, 1+num_drafts-1)
        #     draft_offset += num_drafts-1
        # else:
        #     draft_generated_tokens = draft_generate(
        #         draft_model, token, draft_offset, draft_kv_cache, num_drafts
        #     ) # draft_generated_tokens.shape: (1, num_drafts)
        #     draft_offset += num_drafts

        draft_generated_tokens = draft_generate(
            draft_model, token, draft_offset, draft_kv_cache, num_drafts
        ) # draft_generated_tokens.shape: (1, num_drafts)
        draft_offset += num_drafts

        # Same tokenizer for both models
        draft_generated_tokens = torch.concat([token, draft_generated_tokens], dim=-1)
        new_tokens = step(model, draft_generated_tokens, offset, kv_cache, n_tokens=num_drafts+1)
        offset += num_drafts+1
        advance_new_token = new_tokens[..., -1:] # (1, 1)
        new_tokens = torch.concat([token, new_tokens[..., :-1]], dim=-1)
        assert new_tokens.shape == draft_generated_tokens.shape and new_tokens.shape == (1, num_drafts+1)
        assert draft_offset == offset - 1

        all_correct = True
        for i in range(num_drafts+1):
            if new_tokens[0, i] == draft_generated_tokens[0, i]: continue 
            
            assert i >= 1
            revert_len = num_drafts - i
            revert_cache(draft_kv_cache, revert_len)
            draft_offset -= revert_len
            revert_cache(kv_cache, revert_len+1)
            offset -= revert_len+1

            all_correct = False
            token = new_tokens[i] # (1, 1)
            assert offset == draft_offset
            assert offset == kv_cache[0].offset
            break

        if not all_correct: continue 
        draft_generate(draft_model, draft_generated_tokens[...,-1:], draft_offset, draft_kv_cache, 1)
        token = advance_new_token
        draft_offset += 1

# "I am very happy to"
# [0.8321831, 0.3123, ...] []

# [[[0.2, 0.6, 0.8], [0.2321, 0.9123, 0.7], ... , [0.9, 0.95, 0.3]]]

# # 0.8 -> idx(0) -> am 
# # [[0.8, 0.1, 0.05, ..., x100], [], [], [], []]

# [[0], [99], [15], [23], [5]]

# ha ppy 

# ["I am very happy to"] 
# CPU Core1: "I am very happy to" -> "see"
# CPU Core2: "I am very happy to see" -> "you"
# CPU Core3: "I am very happy to see you" -> "again"

# CPU Core5: 
# ...
# CPU Core10: ...
# # ["I am very happy to | see you again"]

# Batch, num_tokens, emb_dim

# # [["I"], ..., ["I am very happy"]] 
# # [["I"], ..., ["I am very happy"], ["I am very happy to"]]
# # [["I"], ..., ["I am very happy"], ["I am very happy to"], ["I am very happy to see"]]

# # [["I"], ..., ["I am very happy to"], ["I am very happy to see"], ["I am very happy to see you"]]

# [..., 0.9, 0.95, 0.3]

@torch.no_grad() # torch.inference_mode()
def avg_logprob_answer(model, tokenizer, prompt, answer, device="cpu"):
    prompt_ids = tokenizer.encode(prompt) # python list[int], length == L_p
    answer_ids = tokenizer.encode(answer) # python list[int], length == L_a
    full_ids = torch.tensor(prompt_ids + answer_ids,
                            device=device, dtype=torch.long) # (L,) L = L_p + L_a

    full_ids_batched = full_ids.unsqueeze(0) # (B, L) B == 1

    logits = model(full_ids_batched) # (B, L, V)
    logits = logits.squeeze(0) # (L, V)

    logprobs = torch.log_softmax(logits, dim=-1) # (L, V)

    start = len(prompt_ids) - 1 
    end = full_ids.shape[0] - 1 # == L-1

    t_idx = torch.arange(start, end, device=device) # (L_a,)
    next_tokens = full_ids[start + 1 : end + 1] # (L_a,)

    next_token_logps = logprobs[t_idx, next_tokens] # (L_a,) advanced indexing of (t_idx[i], next_tokens[i])

    return next_token_logps.mean().item()

@torch.no_grad()
def avg_logprob_masked(model, 
                       input_ids, mask, # Both L
                       device):
    input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.long) # (L,)
    mask = torch.as_tensor(mask, device=device) # (L,) with values

    input_ids_batched = input_ids.unsqueeze(0) # (B, L) B==1

    logits = model(input_ids_batched) # (B, L, V)
    logits = logits.squeeze(0) # (L, V)

    logprobs = torch.log_softmax(logits, dim=-1) # (L, V)

    pos = torch.nonzero(mask == 1, as_tuple=False).squeeze(1) # (L_m, 1)
    pos = pos.squeeze(-1) # (L_m, 1)
    pos = pos[pos < input_ids.shape[0]] # (L_m',) L_m' might be == L_m or == L_m-1
    tok = input_ids[pos+1] # (L_m',)
    selected_logps = logprobs[pos+1, tok] # (L_m',)

    return torch.mean(selected_logps).item()
