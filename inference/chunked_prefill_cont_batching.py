import torch
from inference.kv_cache import FullKvCache, BatchingKvCache
from inference.generate import step

class InferenceRequest:
    def __init__(
        self,
        model,
        prompt_token_ids: torch.Tensor,
        *,
        request_idx: int,
        prefill_max_step: int = 128,
        eos_token_id,
    ):
        self.kv_cache = [FullKvCache() for _ in range(len(model.transformer_blocks))]
        self.model = model
        self.prompt_token_ids = prompt_token_ids # (1, num_prompt_tokens)
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.next_token = None
        self.offset = 0
        self.request_idx = request_idx
        self.output_indices = torch.empty(1, 0, dtype=torch.float32) # (1, num_prompt_tokens+num_total_tokens_generated)
        self.eos_token_id = eos_token_id

    def try_prefill(self):
        assert not self.is_prefill_done

        tokens_to_prefill = min(self.prefill_max_step, self.prompt_token_ids.numel() - self.offset)
        token = step(
            self.model,
            self.prompt_token_ids[...,self.offset : self.offset + tokens_to_prefill],
            self.offset,
            self.kv_cache,
        )
        self.offset += tokens_to_prefill
        if self.offset >= self.prompt_token_ids.numel():
            self.is_prefill_done = True
            self.decode_done(token, update_offset=False)

    def decode_done(self, token, update_offset=True):
        assert not self.is_done
        
        if token == self.eos_token_id:
            self.is_done = True
            return
        
        self.next_token = token
        # self.output_indices = token if self.output_indices is None else torch.cat(
        #     (self.output_indices, token), dim=-1
        # )
        self.output_indices = torch.cat((self.output_indices, token), dim=-1)
        self.offset += 1 if update_offset else 0
    
def batch_generate(
    model: any,
    prompt_list: list[torch.Tensor], # list of (seq_len, ) or (1, seq_len)
    max_seq_len=4096,
    batch_size=6,
    prefill_step_size=128
):
    is_idle = [True] * batch_size
    decode_requests: list[InferenceRequest] = [None] * batch_size

    batch_kv_cache_all_layers = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(len(model.transformer_blocks))
    ]
    results = []
    pending_prefill_request = None

    while True:
        if len(prompt_list) == 0 and all(is_idle):
            break

        if len(prompt_list) > 0 and pending_prefill_request is None:
            tok_indices = prompt_list.pop(-1)
            pending_prefill_request = InferenceRequest(
                model, tok_indices, prefill_step_size=prefill_step_size, request_idx=len(prompt_list)
            )

        # Chunked prefill
        if pending_prefill_request is not None:
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill() # Will Prefill in chunks
            else:
                prefill_kv_cache = pending_prefill_request.kv_cache
                for i in range(batch_size):
                    if not is_idle[i]: continue

                    is_idle[i] = False
                    # Add kv_cache to each layer's batch_cache
                    for prefill_cache, batch_cache in zip(
                        prefill_kv_cache, batch_kv_cache_all_layers
                    ):
                        batch_cache.add_request(prefill_cache, i)

                    decode_requests[i] = pending_prefill_request
                    pending_prefill_request = None
                    break                   

        # Continuous batching
        if all(is_idle):
            continue

        next_tokens = [req.next_token if req else 0 for req in decode_requests]
        offsets = [req.offset if req else 0 for req in decode_requests]
        next_tokens = torch.concat(next_tokens, dim=0) # (batch_size, 1)

        next_tokens = step(model, next_tokens, offsets, batch_kv_cache_all_layers, n_tokens=1) # (B, 1)
        for i in range(batch_size):
            if not is_idle[i]:
                req = decode_requests[i]
                to_remove_request = req.is_done or req.offset >= max_seq_len
                if not to_remove_request: 
                    req.decode_done(next_tokens[i:i+1,...])
                else:
                    print(f"Removing request {i} b/c it is done or over max len", flush=True)
                    batch_cache.remove_request(i)
                    is_idle[i] = True
                    decode_requests[i] = None
                    results.append((req.request_idx, req.output_indices))
                continue

    return results
