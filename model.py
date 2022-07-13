import pytorch_lightning as pl
import torch
from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.cuda.amp import autocast, GradScaler

class MusicPerformer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        ### Hyperparameters for the Transformer
        self.transformer = PerformerLM(
            num_tokens = 278,               # vocab size
            max_seq_len = 512,             # max sequence length
            dim = 256,                      # dimension
            depth = 12,                     # layers
            heads = 8,                      # heads
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

        self.transformer = AutoregressiveWrapper(self.transformer)#.to("cuda:0")
    def forward(self, x):
        return self.transformer(x.long(), return_loss = True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self(x)
        return loss

    def validation_step(self,  batch, batch_idx):
        if batch_idx == 0:
            x = batch[0][1][:20]

            sample = self.transformer.generate(start_tokens= x.long(), seq_len= 100, repetition_penalty=0.2)
            #print([vocab[int(i)] for i in sample.tolist()[:30]])
            out = [[int(i) for i in  x.tolist() + sample.tolist()]]
            out = self.trainer.datamodule.tokenizer.tokens_to_midi(out)
            out.dump("outputs/1_%s_out.midi" %self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.005)
        
