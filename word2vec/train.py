import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, RMSprop
import torch.nn as nn


from model import W2V_model, W2V_SGNS_model
from w2v_dataloader import CBOW_dataset, SkipGramDataset, SkipGramNegativeSamplingDataset
from test_embeddings import test_embedding_question_words
import os

import argparse
from tqdm import tqdm
import numpy as np
import json


class W2VTrainer:

    def __init__(self, opt):
        """
        Given the parameters initializes training class
        input:
            opt: Argparse Options
        """
        # get the data
        assert os.path.isfile(opt.dataset_path), "Bad file path!!"

        # Training device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print("Training on {}".format(self.device))

        # Choose the dataset
        print("Loading dataset")
        self.dataset = CBOW_dataset(opt.dataset_path) if opt.dataset_type == 'cbow' else SkipGramDataset(opt.dataset_path)
        print("Dataset loaded")
        self.batch_size = opt.batch_size

        # initialize the network
        self.network = W2V_model(len(self.dataset.vocab_set), hidden_dim=opt.hidden_dim)
        self.network.to(self.device)

        # initialize optmizer and pass training parameters
        self.optimizer = Adam(self.network.parameters(), lr=opt.lr)

        # This combines Softmax and NLLLoss
        self.loss = nn.CrossEntropyLoss()

        #testing parameters
        self.test_interval = opt.checkpoint_interval


    def train(self, opt):
        """
        Trains the model for the number of epochs provided
        """

        # Initialize the Dataloader
        train_dataloader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        # Training starts
        for epoch in range(opt.epochs):

            print("Epoch {}/{}".format(epoch + 1, opt.epochs))
            print('-' * 10)

            running_loss = 0.0
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
            print("Loss after epoch {} : {}".format(epoch + 1, epoch_loss))

            if (epoch+1) % opt.checkpoint_interval == 0:
                print("Saving the model after {} epoch".format(epoch + 1))
                torch.save(self.network.state_dict(), "w2v_ckpt_epoch_{}.pth".format(epoch+1))


                # testing for the cosine similarity
                # get the current embedding
                if opt.test_file:
                    embedding_current = self.network.fc1.weight.clone().cpu().transpose(0, 1).detach().numpy()
                    emb_dict = {'vocab_dict':self.dataset.vocab_word_to_idx, 'embedding': embedding_current}

                    cosine_sim = test_embedding_question_words(emb_dict, opt.test_file)
                    print("Avg cosine similarity after {} epoch - {}".format(epoch+1, cosine_sim))

            print("="*20)

        return 0

    def save_embeddings(self):
        '''
        Saves the word embeddings. Its the weights of the first linear layer.
        A JSON file will be saved as dictionary.
        Dictionary has vocabulary and embeddings.
        '''

        # Get the first layer weights
        embedding = self.network.fc1.weight.cpu().detach().numpy().transpose()

        # Convert it to list
        embedding = embedding.tolist()

        # Create embedding dictionary
        emb_dict = {'vocab_dict':self.dataset.vocab_word_to_idx, 'embedding':embedding}
        with open('embeddings.json', 'w') as fp:
            json.dump(emb_dict, fp)
        print("Embedding saved successfully")

class W2V_SGNS_Trainer:

    def __init__(self, opt):

        assert os.path.isfile(opt.dataset_path), "Bad file path!!"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        print("Training on {}".format(self.device))

        print("Loading the dataset")
        self.dataset = SkipGramNegativeSamplingDataset(opt.dataset_path,
                       k=opt.neg_sample_size, freq_power=0.75)
        print("Dataset Loaded")
        self.batch_size = opt.batch_size

        # initialize the network
        self.network = W2V_SGNS_model(len(self.dataset.vocab_set), hidden_dim=opt.hidden_dim)
        self.network.to(self.device)

        # initialize optimizer and pass training parameters
        self.optimizer = Adam(self.network.parameters(), lr=opt.lr)

        # Binary cross entropy loss to handle two classes
        self.loss = nn.BCELoss()

        #testing parameters
        self.test_interval = opt.checkpoint_interval

    def train(self, opt):
        """
        Trains the model for the number of epochs provided
        """

        train_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        for epoch in range(opt.epochs):

            print("Epoch {}/{}".format(epoch + 1, opt.epochs))
            print('-' * 10)

            running_loss = 0.0
            pbar = tqdm(train_dataloader)

            for batch_i, (input_idxs, context_idxs, targets) in enumerate(pbar):

                input_idxs = input_idxs.view(-1,).to(self.device)
                context_idxs = context_idxs.view(-1,).to(self.device)
                targets = targets.view(-1, 1).to(self.device)

                y_pred = self.network(input_idxs, context_idxs)
                loss = self.loss(y_pred, targets)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()

                pbar.set_description("Loss = %f" % loss.item())

            epoch_loss = running_loss / len(train_dataloader)
            print("Loss after epoch {} : {}".format(epoch + 1, epoch_loss))

            if (epoch+1) % opt.checkpoint_interval == 0:
                if not os.path.exists('checkpoint'):
                    os.mkdir('checkpoint')
                print("Saving the model after {} epoch".format(epoch + 1))
                torch.save(self.network.state_dict(), "checkpoint/w2v_sgns_ckpt_epoch_{}.pth".format(epoch+1))

                #testing for the cosine similarity
                #get the current embedding
                if opt.test_file:
                    embedding_current = self.network.embedding.weight.cpu().detach().numpy()
                    emb_dict = {'vocab_dict':self.dataset.vocab_word_to_idx, 'embedding': embedding_current}

                    cosine_sim = test_embedding_question_words(emb_dict, opt.test_file)
                    print("Avg cosine similarity after {} epoch - {}".format(epoch+1, cosine_sim))

            print("="*20)

        return 0

    def save_embeddings(self):
        '''
        Saves the word embeddings. Its the weights of the first linear layers.
        A JSON file will be saved as dictionary.
        Dictionary has vocabulary and embeddings.
        '''

        # Get the embedding matrix
        embedding = self.network.embedding.weight.cpu().detach().numpy()
        embedding = embedding.tolist()
        emb_dict = {'vocab_dict':self.dataset.vocab_word_to_idx, 'embedding':embedding}
        with open('embeddings_sgns.json', 'w') as fp:
            json.dump(emb_dict, fp)
        print("Embedding saved successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="sgns", help="model type vanilla/sgns")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=300, help="size of the hidden layer dimension")
    parser.add_argument("--dataset_type", type=str, default="sgram", help="cbow or skipgram model")
    parser.add_argument("--batch_size", type=int, default=128, help="size of each word batch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of the model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the JSON dataset")
    parser.add_argument("--neg_sample_size", type=int, default=5, help="Number of negative samples")
    parser.add_argument("--test_file", type=str, help="path to the file used in the testing part")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="path to the training checkpoint")
    opt = parser.parse_args()
    print("=" * 10, "HYPERPARAMETERS", "=" * 10)
    print(opt)

    if opt.model_type == 'sgns':
        trainer = W2V_SGNS_Trainer(opt)
    else:
        trainer = W2VTrainer(opt)

    if opt.checkpoint_path:
        trainer.network.load_state_dict(torch.load(opt.checkpoint_path))

    trainer.train(opt)
    trainer.save_embeddings()

