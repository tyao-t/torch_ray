import torch
from inference.kv_cache import FullKvCache
from generate import prefill, step, revert_cache

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