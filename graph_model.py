import pytorch_lightning as pl
import torch
from performer_pytorch.performer_pytorch import PerformerLM, exists
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p
from networkx.algorithms.clique import number_of_cliques
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool

class GraphPerformerLM(PerformerLM):
    def __init__(self, *, num_tokens, max_seq_len, dim, depth, heads, dim_head=64, local_attn_heads=0, local_window_size=256, causal=False, ff_mult=4, nb_features=None, feature_redraw_interval=1000, reversible=False, ff_chunks=1, ff_glu=False, emb_dropout=0, ff_dropout=0, attn_dropout=0, generalized_attention=False, kernel_fn=..., use_scalenorm=False, use_rezero=False, cross_attend=False, no_projection=False, tie_embed=False, rotary_position_emb=True, axial_position_emb=False, axial_position_shape=None, auto_check_redraw=True, qkv_bias=False, attn_out_bias=False, shift_tokens=False):
        super().__init__(num_tokens=num_tokens, max_seq_len=max_seq_len, dim= dim, depth = depth, heads=heads, dim_head=dim_head, local_attn_heads, local_window_size, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, ff_glu, emb_dropout, ff_dropout, attn_dropout, generalized_attention, kernel_fn, use_scalenorm, use_rezero, cross_attend, no_projection, tie_embed, rotary_position_emb, axial_position_emb, axial_position_shape, auto_check_redraw, qkv_bias, attn_out_bias, shift_tokens)
        self.graph_layer = GCN(hidden_channels=dim)

    def forward(self, x, return_encodings = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embeddings
        x = self.graph_layer(x)
        x += self.pos_emb(x)

        x = self.dropout(x)

        # performer layers

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.performer(x, pos_emb = layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t() ### quesion is here 

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        num_node_features=12
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, edge_dim=13)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=13)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=13)
        #self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        
        return x