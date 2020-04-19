import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.optim import Adam, RMSprop
import torch.nn as nn


from model import W2V_model
from w2v_dataloader import CBOW_dataset, SkipGramDataset
import os

import argparse
from tqdm import tqdm
import numpy as np
import json
import ipdb


class W2VTrainer:

    def __init__(self, opt):
        """
        Given the parameters initializes a CBOW style training class
        input:
            hidden_dim : integer representing the number of nodes in the hidden layer
            path_to_json : String containing the path to the json file containing the
                            data to be used to train.
            l_rate : Float containing the learning rate
            batch_size : The batch size
        """
        # get the data
        assert os.path.isfile(opt.dataset_path), "Bad file path!!"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.dataset = CBOW_dataset(opt.dataset_path) if opt.dataset_type == 'cbow 'else SkipGramDataset(opt.dataset_path)
        self.batch_size = opt.batch_size

        # initialize the network
        self.network = W2V_model(len(self.dataset.vocab_set), hidden_dim=opt.hidden_dim)
        self.network.to(self.device)

        # initialize training parameters
        self.optimizer = RMSprop(self.network.parameters(), lr=opt.lr)
        self.loss = nn.CrossEntropyLoss()

    def train(self, opt):
        """
        Trains the model for the number of epochs provided
        """
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(opt.validation_split * dataset_size))
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)

        for epoch in range(opt.epochs):

            epoch_loss_list = []
            running_loss = 0.0

            print("Epoch {}/{}".format(epoch + 1, opt.epochs))
            print('-' * 10)

            pbar = tqdm(train_dataloader)
            for batch_i, (input_samples, output_samples) in enumerate(pbar):

                input_samples = input_samples.to(self.device).type(torch.float)
                output_samples = output_samples.to(self.device).type(torch.long)

                y_pred = self.network(input_samples)
                loss = self.loss(y_pred, output_samples)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()

                pbar.set_description("Loss = %f" % loss.item())

            epoch_loss = running_loss / len(train_dataloader)
            epoch_loss_list.append(epoch_loss)
            print("Loss after epoch {} : {}".format(epoch + 1, epoch_loss))

            if (epoch+1) % opt.checkpoint_interval == 0:
                print("Saving the model after {} epoch".format(epoch + 1))
                torch.save(self.network.state_dict(), "w2v_ckpt_epoch_{}.pth".format(epoch+1))
            print("="*20)

        return 0

    def save_embeddings(self):
        embedding = self.network.fc1.weight.cpu().transpose(0, 1).detach().numpy()
        embedding = embedding.tolist()
        emb_dict = {'vocab_dict':self.dataset.vocab_word_to_idx, 'embedding':embedding}
        with open('embeddings.json', 'w') as fp:
            json.dump(emb_dict, fp)
        print("Embedding saved successfully")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=300,
                        help="size of the hidden layer dimension")
    parser.add_argument("--dataset_type", type=str, default="cbow", help="cbow or skipgram model")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each word batch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of the model")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="interval between saving model weights")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the JSON dataset")
    parser.add_argument("--validation_split", type=float,
                        default=0.1, help="Dataset validation split")
    opt = parser.parse_args()
    print("=" * 10, "HYPERPARAMETERS", "=" * 10)
    print(opt)

    trainer = W2VTrainer(opt)

    trainer.network.load_state_dict(torch.load('w2v_ckpt_epoch_18.pth'))
    # trainer.train(opt)
    ipdb.set_trace()
    # trainer.save_embeddings()
