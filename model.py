import pytorch_lightning as pl
import torch
from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper, top_k, top_p

class MusicPerformer(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        ### Hyperparameters for the Transformer
        self._hparams = hparams
        self.transformer = PerformerLM(
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

        self.transformer = AutoregressiveWrapper(self.transformer)#.to("cuda:0")
    def forward(self, x):
        return self.transformer(x.long(), return_loss = True)

    def training_step(self, batch, batch_idx):
        x = batch[0] 
        loss = self(x)
        return loss

    def validation_step(self,  batch, batch_idx):
        if batch_idx == 0:
            self.generate_sequence(batch[0][1], output_filename=self.current_epoch,
                                   num_tokens_to_use=self._hparams["init_len_generate"],
                                   num_token_to_generate= self._hparams["seq_len_generate"], 
                                   repetition_penalty=self._hparams["repetition_penalty"],
                                   temperature=self._hparams["temperature"])

    def generate_sequence(self, input_sequence, output_filename: str, num_tokens_to_use: int = 50,
                                num_token_to_generate: int = 100,
                                temperature = 1.0, 
                                repetition_penalty = 0.0,
                                filter_thres = 0.9, 
                                filter_logits_fn="top_k"):
        if filter_logits_fn == "top_k":
            filter_logits_fn = top_k
        elif filter_logits_fn == "top_p":
            filter_logits_fn = top_p
        else:
            raise NotImplementedError()

        x = input_sequence[:num_tokens_to_use]
        out = self.transformer.generate(start_tokens= x.long(), 
                                        seq_len=num_token_to_generate, 
                                        repetition_penalty=repetition_penalty,
                                        temperature=temperature, 
                                        filter_thres=filter_thres,
                                        filter_logits_fn=filter_logits_fn)

        out = [[int(i) for i in  x.tolist() + out.tolist()]]
        out = self.trainer.datamodule.tokenizer.tokens_to_midi(out)
        out.dump("outputs/%s_%s.midi" %(self._hparams["version"], output_filename))
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self._hparams["learning_rate"], 
                                weight_decay=self._hparams["weight_decay"])



