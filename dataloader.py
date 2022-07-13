import json
import pytorch_lightning as pl
from miditok import REMI
from pathlib import Path
from torch.utils.data import random_split, DataLoader, TensorDataset
import glob
import torch


class MidiDataModule(pl.core.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self._hparams = hparams
        self.tokenizer = self.init_tokenizer(hparams["tokenizer"])

    def prepare_data(self):
        ### Convert MIDI files to JSON
        if self._hparams["prepare_data"]:
            self.tokenizer.tokenize_midi_dataset(self.get_path("train_midi"),
                     "data/train", self.midi_valid)
            self.tokenizer.tokenize_midi_dataset(self.get_path("val_midi"),
                     "data/val", self.midi_valid)

    def setup(self, stage: str = None):
        self.train_dataset = self.get_dataset(dir="train")
        self.val_dataset = self.get_dataset(dir="val")
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._hparams["batch_size"], 
                          num_workers=self._hparams["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._hparams["batch_size"],
                          num_workers=self._hparams["num_workers"])
    
    def init_tokenizer(self, tokenizer = "REMI"):
        if tokenizer == "REMI":
            return REMI(pitch_range=self._hparams["pitch_range"], 
                        beat_res = self._hparams["beat_res"] , 
                        nb_velocities = self._hparams["nb_velocities"], 
                        additional_tokens = self._hparams["additional_tokens"], 
                        mask=True)
        else:
            raise NotImplementedError()

    def get_dataset(self, dir: str):
        max_len = self._hparams["sequence_len"]
        files = glob.glob(self._hparams["data_path"] + "/%s/*.json" %dir)
        x = torch.zeros(size=(len(files), max_len))
        for i, file in enumerate(files):
            with open(file, "rb") as f:
                _x = torch.tensor(json.load(f)["tokens"][0][:max_len]) 
                x[i, :len(_x) ] += _x
        return TensorDataset(x)

    def get_path(self, dir: str):
        return list(Path(self.data_path, dir).glob("*.midi"))

    @staticmethod
    def midi_valid(midi) -> bool:
        if any(ts.numerator != 4 for ts in midi.time_signature_changes):
            return False  # time signature different from 4/*, 4 beats per bar
        if midi.max_tick < 10 * midi.ticks_per_beat:
            return False  # this MIDI is too short
        return True
