import torch 
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler 
from torch.optim import Adam
import torch.nn as nn


from model import W2V_model
from w2v_dataloader import CBOW_dataset
import os

from tqdm import tqdm
import numpy as np
import pdb
class CBOW:

    def __init__(self, hidden_dim, path_to_json, l_rate=0.0001,
                 batch_size=64, checkpoint_interval=None,
                 ):
        """
        Given the parameters initializes a CBOW style training class
        input:
            hidden_dim : integer representing the number of nodes in the hidden layer
            path_to_json : String containing the path to the json file containing the 
                            data to be used to train.
            l_rate : Float containing the learning rate
            batch_size : The batch size
        """
        #get the data
        assert os.path.isfile(path_to_json), "Bad file path!!"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 'cpu')

        self.dataset = CBOW_dataset(path_to_json)

        self.batch_size = batch_size
        #initialize the network
        self.network = W2V_model(len(self.dataset.vocab_set), hidden_dim=hidden_dim)
        self.network = self.network.to(self.device)

        #initialize training parameters
        self.optimizer = Adam(self.network.parameters(), lr=l_rate)

        self.loss = nn.CrossEntropyLoss()
        #initialize other hyperparameters
        self.batch_size = batch_size





    def train(self, epochs, validation_split=0.5):
        """
        Trains the model for the number of epochs provided
        """        
        dataset_size = self.dataset.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.batch_size,
                                      shuffle=True)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)

        for i in tqdm(range(epochs)):

            epoch_loss = []


            for batch_i, (input_samples, output_samples) in enumerate(train_dataloader):

                input_samples = input_samples.to(self.device).type(torch.float)
                output_samples = output_samples.to(self.device).type(torch.long)
                

                y_pred = self.network(input_samples)
                loss = self.loss(y_pred, output_samples)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(loss)

        return 0



if __name__ == '__main__':
    
    cbow_test = CBOW(300, 'cbow_style_training_dataset.json')
    cbow_test.train(20)



