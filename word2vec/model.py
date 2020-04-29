import torch
import torch.nn as nn
import numpy as np


class W2V_model(nn.Module):
    """This class represents the Word2Vec model"""

    def __init__(self, vocab_size, hidden_dim=300):
        super(W2V_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.fc1 = nn.Linear(vocab_size, hidden_dim) #projection layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) #hidden layer   
        self.fc3 = nn.Linear(hidden_dim, vocab_size) #output layer

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
