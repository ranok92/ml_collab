import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import utils as ut
import os

class CBOW_dataset(Dataset):
    """CBOW dataset """

    def __init__(self, cbow_json_path):
        """
        Args:
            cbow_json_path : The CBOW JSON dataset path
        """
        self.vocab_word_to_idx = {}
        self.vocab_set = set()

        with open(cbow_json_path) as f:
            self.json_data = json.load(f)
            print("Building vocabulary")
            for ith_words in tqdm(self.json_data['dataset']):
                output, inputs = ith_words[-1], ith_words[:-1]
                self.vocab_set.add(output)
                self.vocab_set.update(inputs)

        # Vocabulary list
        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

    def __len__(self):
        return len(self.json_data['dataset'])

    def __getitem__(self, idx):
        words = self.json_data['dataset'][idx]
        output, inputs = words[-1], words[:-1]

        # We need output class as PyTorch makes one hot internally
        output_class = self.vocab_word_to_idx[output]

        # Get one hot for all inputs
        inputs_ohv = sum([ut.get_one_hot_vec(input, self.vocab_word_to_idx) for input in inputs])
        return inputs_ohv, output_class


class SkipGramDataset(Dataset):
    """Skip Gram Dataset"""

    def __init__(self, skipgram_json_path):
        """
        Args:
            skipgram_json_path : The SkipGram JSON dataset path
        """
        assert os.path.isfile(skipgram_json_path), "File does not exist."

        self.vocab_word_to_idx = {}
        self.vocab_set = set()

        with open(skipgram_json_path) as f:
            self.json_data = json.load(f)
            print("Building vocabulary")
            for ith_words in tqdm(self.json_data['dataset']):
                input, output = ith_words[0], ith_words[1]
                self.vocab_set.add(input)
                self.vocab_set.add(output)

        # Vocabulary list
        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

    def __len__(self):
        return len(self.json_data['dataset'])

    def __getitem__(self, idx):
        words = self.json_data['dataset'][idx]
        input, output = words[0], words[1]
        input_ohv = ut.get_one_hot_vec(input, self.vocab_word_to_idx)
        output_class = self.vocab_word_to_idx[output]
        return input_ohv, output_class


class SkipGramNegativeSamplingDataset(Dataset):
    """Skip Gram Neagtive Sampling Dataset"""

    def __init__(self, skipgram_json_path, k=5,
                 sample_pool_size=5000000,
                 freq_power=1):
        """
        Args:
            skipgram_json_path : The path to a json dictionary containing
                                the dataset and the frequency dictionary.

                                {'dataset': dataset, 'freq_dict':frequency dictionary}
            sample_pool_size : The number of samples selected according to the frequency
                                distribution from which the uniform sampling will take
                                place.
            freq_power : The power to which each of the word frequencies are raised before
                         generating the sampling distribution.
        """
        self.k = k
        self.distribution_sample_size = sample_pool_size
        self.freq_power = freq_power
        self.vocab_word_to_idx = {}
        self.vocab_set = set()
        assert os.path.isfile(skipgram_json_path), "File does not exist."

        with open(skipgram_json_path) as f:
            self.json_data = json.load(f)

            dataset = self.json_data['dataset']
            print("Building Vocabulary")
            for ith_words in tqdm(dataset):
                inp, output = ith_words[0], ith_words[1]
                self.vocab_set.add(inp)
                self.vocab_set.add(output)

        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

        # Get the frequency of words
        self.word_freq_distribution = [self.json_data['freq_dict'][word] for word in self.vocab_set]
        self.word_freq_distribution = np.power(self.word_freq_distribution, self.freq_power)
        total_freq = sum(self.word_freq_distribution)

        # Probability of each word
        self.word_freq_prob = [x/total_freq for x in self.word_freq_distribution]

        # Generate word indices according to the probability distribution
        # self.sampling_distribution = np.random.choice(len(self.word_freq_prob),size=self.distribution_sample_size, p=self.word_freq_prob)
        self.sampling_distribution = torch.multinomial(torch.tensor(self.word_freq_prob),
                                                       self.distribution_sample_size,
                                                       replacement=True)
        print("Sampling distribution Created")


    def __len__(self):
        return len(self.json_data['dataset'])


    def __getitem__(self, idx):
        input_idxs, output_idxs, targets = [], [], []

        # get the ith word tuple from dataset
        words = self.json_data['dataset'][idx]
        input, output = words[0], words[1]
        input_idx = self.vocab_word_to_idx[input]
        output_idx = self.vocab_word_to_idx[output]

        # Add the positive tuple to input and output
        input_idxs.append(input_idx)
        output_idxs.append(output_idx)

        # Add the positive target
        targets.append(1)

        # Add the negative samples
        input_idxs.extend([input_idx] * self.k)

        # for word frequency based distribution
        neg_idxs = []
        for _ in range(self.k):
            while True:
                # idx = self.sampling_distribution[np.random.randint(self.distribution_sample_size)]
                idx = np.random.randint(self.distribution_sample_size)
                idx = self.sampling_distribution[idx].item()
                if (idx not in [input_idx, output_idx]) and (idx not in neg_idxs):
                    neg_idxs.append(idx)
                    break

        output_idxs.extend(neg_idxs)

        # Add the negative targets
        targets.extend([0] * self.k)

        # Reshape the target
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
        targets = torch.from_numpy(targets)
        return torch.tensor(input_idxs), torch.tensor(output_idxs), targets

