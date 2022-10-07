import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import pandas as pd
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class EarlyStopping():

    # CHANGED PAT FROM 5 TO 2 AND REMOVED THE COUNTER=0
    def __init__(self, patience=30, min_delta=0.0):  #previously 30, with 200eps was good
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = 9999
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == 9999:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            #self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True



class load_fold(Dataset):
    def __init__(self, filename, datadir):

        self.filename = filename
        self.datadir = datadir
        xy = pd.read_csv(filename,header=None)

        self.file = xy.values[:,0]
        self.lab1 = xy.values[:,1]
        

        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):

        return self.file[idx], self.lab1[idx]*0.1



class self_attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(True),
                nn.Linear(64, 1)
        )
        self.layer_norm = nn.LayerNorm(100)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, encoder_outputs):
        
        energy = torch.matmul(encoder_outputs,encoder_outputs.transpose(1,2))
        weights = F.softmax(energy, dim=1)
        outputs = torch.matmul(weights,encoder_outputs)
        outputs = self.dropout(outputs)
        
        outputs = self.layer_norm(outputs)
        
        return outputs, weights




class TRANSFORMER(nn.Module):
    def __init__(self, n_gru_layers, hidden_dim, batch_size):
        super(TRANSFORMER,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_gru_layers = n_gru_layers
        self.batch_size = batch_size

        self.gru = nn.GRU(40, 100, self.n_gru_layers,batch_first=True, bidirectional=True)
        self.gru_final = nn.GRU(100,50, self.n_gru_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.hidden_dim,self.hidden_dim)
        #self.fc = nn.Linear(40,40)
        self.fc_final = nn.Linear(self.hidden_dim,1)
        #self.fc2 = nn.Linear(int(self.hidden_dim/2),1)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.5)   #0.5
        self.sigmoid = nn.Sigmoid()
        
        self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim) #, affine=False)
        self.bn = nn.BatchNorm1d(num_features=self.hidden_dim)

        self.attention = self_attention(hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, input_, seq_len):

        output, h1 = self.gru(input_) 
        seq_unpacked, len_unpacked = pad_packed_sequence(output,batch_first=True)
        seq_unpacked = seq_unpacked[:,:,:self.hidden_dim]+seq_unpacked[:,:,self.hidden_dim:]

        outputs, attn_weights = self.attention(seq_unpacked) 
        
        x1 = self.fc(outputs)
        x1 = self.relu(x1)
        
        x1 = self.fc(x1)
        x1 = self.relu(x1)
        
        x1 = self.fc(x1)
        x1 = self.relu(x1)
        outputs1 = x1 + outputs        


        output_final1 = F.max_pool2d(outputs1, kernel_size=outputs.size()[1:])
        output_final1 = self.relu(output_final1)
        
        
        return output_final1, attn_weights


