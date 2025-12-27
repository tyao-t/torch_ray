import torch
import torch.nn as nn

class VectorizedLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_all_gates = nn.Linear(input_sz + hidden_dim, hidden_dim * 4)

    def forward(self, x, init_states=None):
        B, L, d_in = x.size()
        
        if init_states is None:
            hidden_state = torch.zeros(B, self.hidden_dim).to(x.device)
            cell_state = torch.zeros(B, self.hidden_dim).to(x.device)
        else:
            hidden_state, cell_state = init_states
            
        outputs = []
        
        for t in range(L):
            input_at_t = x[:, t, :] # (B, d_in)
            concat_input_hidden = torch.cat((hidden_state, input_at_t), dim=-1) # (B, d_in)
    
            gates_out = self.weight_all_gates(concat_input_hidden) # (B, 4 * hidden)
            
            forget_logit, input_logit, cell_logit, output_logit = gates_out.chunk(4, dim=-1) # (B, hidden) each
            
            forget_gate = torch.sigmoid(forget_logit) # (B, hidden)
            input_gate = torch.sigmoid(input_logit) # (B, hidden)
            cell_candidate = torch.tanh(cell_logit) # (B, hidden)
            output_gate = torch.sigmoid(output_logit) # (B, hidden)
            
            cell_state = forget_gate * cell_state + input_gate * cell_candidate # (B, hidden)
            
            hidden_state = output_gate * torch.tanh(cell_state) # (B, hidden)

            outputs.append(hidden_state.unsqueeze(-2))

        return torch.cat(outputs, dim=-2), (hidden_state, cell_state) # (B, T, hidden), ((B, hidden), (B, hidden))

class VectorizedGRU(nn.Module):
    def __init__(self, input_sz: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_reset_update = nn.Linear(input_sz + hidden_dim, hidden_dim * 2)
        self.weight_candidate = nn.Linear(input_sz + hidden_dim, hidden_dim)

    def forward(self, x, init_states=None):
        B, L, d_in = x.size()
        
        if init_states is None:
            hidden_state = torch.zeros(B, self.hidden_dim).to(x.device)
        else:
            hidden_state = init_states
            
        outputs = []
        
        for t in range(L):
            input_at_t = x[:, t, :] # (B, d_in)
            
            concat_for_gates = torch.cat((hidden_state, input_at_t), dim=-1) # (B, d_in + hidden)
            
            gates_out = self.weight_reset_update(concat_for_gates) # (B, 2 * hidden)
            reset_logit, update_logit = gates_out.chunk(2, dim=-1) # (B, hidden) each
            
            reset_gate = torch.sigmoid(reset_logit)
            update_gate = torch.sigmoid(update_logit)
            
            concat_for_candidate = torch.cat((reset_gate * hidden_state, input_at_t), dim=-1) # (B, d_in + hidden)
            
            candidate_logit = self.weight_candidate(concat_for_candidate) # (B, hidden)
            candidate_hidden = torch.tanh(candidate_logit)
            
            hidden_state = (1 - update_gate) * hidden_state + update_gate * candidate_hidden # (B, hidden)
            
            outputs.append(hidden_state.unsqueeze(-2))
            
        return torch.cat(outputs, dim=-2), hidden_state # (B, T, hidden), (B, hidden)
