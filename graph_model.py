import pytorch_lightning as pl
import torch
from performer_pytorch.performer_pytorch import PerformerLM, exists
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from model import MusicPerformer

class GraphMusicPerformer(MusicPerformer):
    def __init__(self, hparams) -> None:
        super().__init__()
        self._hparams = hparams
        self.transformer = GraphPerformerLM(
            num_tokens = hparams["vocab_size"],               # vocab size
            max_seq_len = hparams["sequence_len"],             # max sequence length
            dim = hparams["hidden_size"],                      # dimension
            depth = hparams["num_encoders"],                     # layers
            heads = hparams["num_heads"],                      # heads
            causal = True,                 # auto-regressive or not
            nb_features = 256,              # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
            no_projection=False,
            feature_redraw_interval = 1000, # how frequently to redraw the projection matrix, the more frequent, the slower the training
            reversible = True,              # reversible layers, from Reformer paper
            ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
            use_scalenorm = False,          # use scale norm, from 'Transformers without Tears' paper
            use_rezero = True,             # use rezero, from 'Rezero is all you need' paper
            ff_glu = True,                  # use GLU variant for feedforward
            emb_dropout = 0.1,              # embedding dropout
            ff_dropout = 0.1,               # feedforward dropout
            attn_dropout = 0.1,             # post-attn dropout
            local_attn_heads = 4,           # 4 heads are local attention, 4 others are global performers
            local_window_size = 36,        # window size of local attention
            rotary_position_emb = True,     # use rotary positional embedding, which endows linear attention with relative positional encoding with no learned parameters. should always be turned on unless if you want to go back to old absolute positional encoding
            shift_tokens = False)

        self.transformer = AutoregressiveGraphWrapper(self.transformer)#.to("cuda:0")

class AutoregressiveGraphWrapper(AutoregressiveWrapper):
    def forward(self, x, **kwargs):
        #### x: [batch_size = 1, sequence_size, feature_size = 12]
        xi = x[:, :-1] 
        xo = x[:, 1:]

        out = self.net(xi, **kwargs)

        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
class GraphPerformerLM(PerformerLM):
    def __init__(self, num_tokens, max_seq_len, dim, depth, heads, dim_head=64, local_attn_heads=0, local_window_size=256, causal=False, ff_mult=4, nb_features=None, feature_redraw_interval=1000, reversible=False, ff_chunks=1, ff_glu=False, emb_dropout=0, ff_dropout=0, attn_dropout=0, generalized_attention=False, kernel_fn=..., use_scalenorm=False, use_rezero=False, cross_attend=False, no_projection=False, tie_embed=False, rotary_position_emb=True, axial_position_emb=False, axial_position_shape=None, auto_check_redraw=True, qkv_bias=False, attn_out_bias=False, shift_tokens=False):
        super().__init__(num_tokens=num_tokens, max_seq_len=max_seq_len, dim= dim, depth = depth, heads=heads, dim_head=dim_head, 
        local_attn_heads = local_attn_heads,
        local_window_size = local_window_size, 
        causal=causal, 
        ff_mult = ff_mult, 
        nb_features = nb_features, 
        feature_redraw_interval = feature_redraw_interval, 
        reversible = reversible, 
        ff_chunks = ff_chunks, 
        ff_glu = ff_glu, 
        emb_dropout = emb_dropout, 
        ff_dropout = ff_dropout, 
        attn_dropout = attn_dropout, 
        generalized_attention = generalized_attention, 
        kernel_fn = None, 
        use_scalenorm = use_scalenorm, 
        use_rezero = use_rezero, 
        cross_attend = cross_attend, 
        no_projection = no_projection,
        tie_embed = False, 
        rotary_position_emb = rotary_position_emb, 
        axial_position_emb = axial_position_emb, 
        axial_position_shape = axial_position_shape, 
        auto_check_redraw = auto_check_redraw, 
        qkv_bias = qkv_bias, 
        attn_out_bias = attn_out_bias, 
        shift_tokens = shift_tokens)
        self.graph_layer = GCN(hidden_channels=dim)
        del self.token_emb 
        del self.to_out
        self.to_out = Linear(dim, 12)

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

        return self.to_out(x)


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