import networkx as nx
import matplotlib.pyplot as plt
import music21
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.utils import erdos_renyi_graph, to_networkx
from torch_geometric.data import Data

from IPython.display import Image, Audio

from torch.utils.data import ConcatDataset, DataLoader
from torch_geometric.data import Data

max_val = 10
def return_graph(stream):
  stream_c = stream.chordify()
  piece = []
  for element in stream_c.recurse():
    if type(element) == music21.stream.Measure:
      latest_offset = element.offset
    chord = np.ones(max_val+2)*-1
    if type(element) == music21.chord.Chord:
      note_list = []
      for i in range(len(element)):
        note_list.append(element[i].pitch.midi%12)
      note_list.sort()
      for i in range(len(element)):
        chord[i] = note_list[i]
      chord[10] = element.duration.quarterLength
      chord[11] = element.offset + latest_offset
      piece.append(chord)
  piece = np.array(piece)
  x = torch.tensor(piece, dtype=torch.float)
  edge_music = torch.stack([torch.arange(0, x.shape[0] - 1), torch.arange(1, x.shape[0])])

  edge_attr = torch.zeros((x.shape[0] - 1, 3))
  edge_attr[:,0] = 1
  dist_attr = x[1:,:10] - x[:-1,:10]
  edge_attr = torch.cat((edge_attr, dist_attr), dim=1)
  for i in range(x.shape[0]):
    for j in range(x.shape[0]):
      if j != i:
        temp = (piece[i,:4] - piece[j,:4]) % 12
        if np.all(temp == temp[0]):
          transpose_edge = torch.tensor([[i], [j]], dtype=torch.float)
          edge_music = torch.cat((edge_music, transpose_edge), dim=1)
          transpose_attr = torch.tensor([[0,1,0]], dtype=torch.float)
          dist_attr = (x[i,:10] - x[j,:10]).reshape(1,-1)
          
          transpose_attr = torch.cat((transpose_attr, dist_attr), dim=1)
          edge_attr = torch.cat((edge_attr, transpose_attr))
        temp2 = (12 - piece[i,:4] - piece[j,:4])
        if np.all(temp2 == 0):
          inverse_edge = torch.tensor([[i], [j]], dtype=torch.float)
          edge_music = torch.cat((edge_music, inverse_edge), dim=1)
          inverse_attr = torch.tensor([[0,0,1]], dtype=torch.float)
          dist_attr = (x[i,:10] - x[j,:10]).reshape(1,-1)
          transpose_attr = torch.cat((transpose_attr, dist_attr), dim=1)
          edge_attr = torch.cat((edge_attr, inverse_attr))
  data = Data(x=x, edge_index=edge_music.contiguous(), edge_attr=edge_attr)
  edge_index = edge_music.type(dtype=torch.LongTensor)
  return data, x, edge_index, edge_attr

class GraphDataModule(pl.core.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self._hparams = hparams

    def setup(self, stage: str = None):
        datasets = []
        for chorale in music21.corpus.chorales.Iterator(1, returnType='filename'):
            sample = music21.corpus.parse(chorale)
            data, x, edge_music, edge_attr = return_graph(sample)
            datasets.append(data)
    ### what we are missing is some kind of encoding back to the chords? 
        self.train_dataset = ConcatDataset(datasets=datasets)
        self.val_dataset = ConcatDataset(datasets=datasets)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, 
                          num_workers=self._hparams["num_workers"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1,
                          num_workers=self._hparams["num_workers"], shuffle=False)
    

