import pytorch_lightning as pl
import torch
from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper
from torch.cuda.amp import autocast, GradScaler

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


class MusicPerformer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.transformer = PerformerLM(
            num_tokens = 278,
            dim = 256,
            depth = 10,
            max_seq_len = 512,
            heads = 8,
            causal = True,
            reversible = True,
            nb_features = 256,
            use_scalenorm = True,
            shift_tokens = True,
            local_attn_heads = 2,    
            no_projection=False) ##if true - standard SoftMax Attention
            
        self.transformer = AutoregressiveWrapper(self.transformer).to("cuda:0")
    def forward(self, x):
        return self.transformer(x.long(), return_loss = True)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self(x)
        return loss

    def validation_step(self,  batch, batch_idx):
        if batch_idx == 0:
            x = batch[0][1][:20]
        #prime = decode_tokens(x)
        #print(f'%s \n\n %s', (prime, '*' * 100))
            #vocab = self.trainer.datamodule.tokenizer.vocab.token_to_event
            #print([vocab[int(i)] for i in x.tolist()[:30]])
            sample = self.transformer.generate(start_tokens= x.long(), seq_len= 100, repetition_penalty=0.2)
            #print([vocab[int(i)] for i in sample.tolist()[:30]])
            out = [[int(i) for i in  x.tolist() + sample.tolist()]]
            out = self.trainer.datamodule.tokenizer.tokens_to_midi(out)
            out.dump("outputs/1_%s_out.midi" %self.current_epoch)

        #output_str = decode_tokens(sample)
        #print(output_str)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.005)
        
