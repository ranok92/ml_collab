import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import utils as ut


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
            for ith_words in self.json_data:
                output, inputs = ith_words[-1], ith_words[:-1]
                self.vocab_set.add(output)
                self.vocab_set.update(inputs)
        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        words = self.json_data[idx]
        output, inputs = words[-1], words[:-1]
        # output_ohv = ut.get_one_hot_vec(output, self.vocab_word_to_idx)
        output_class = self.vocab_word_to_idx[output]
        inputs_ohv = sum([ut.get_one_hot_vec(
            input, self.vocab_word_to_idx) for input in inputs])
        return inputs_ohv, output_class


class SkipGramDataset(Dataset):
    """Skip Gram Dataset"""

    def __init__(self, skipgram_json_path):
        """
        Args:
            skipgram_json_path : The SkipGram JSON dataset path
        """
        self.vocab_word_to_idx = {}
        self.vocab_set = set()
        with open(skipgram_json_path) as f:
            self.json_data = json.load(f)
            for ith_words in self.json_data:
                input, output = ith_words[0], ith_words[1]
                self.vocab_set.add(input)
                self.vocab_set.add(output)
        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        words = self.json_data[idx]
        input, output = words[0], words[1]
        input_ohv = ut.get_one_hot_vec(input, self.vocab_word_to_idx)
        # output_ohv = ut.get_one_hot_vec(output, self.vocab_word_to_idx)
        output_class = self.vocab_word_to_idx[output]
        return input_ohv, output_class

class SkipGramNegativeSamplingDataset(Dataset):
    """Skip Gram Neagtive Sampling Dataset"""

    def __init__(self, skipgram_json_path, k=5):
        """
        Args:
            skipgram_json_path : The SkipGram JSON dataset path
        """
        self.k = k
        self.vocab_word_to_idx = {}
        self.vocab_set = set()
        with open(skipgram_json_path) as f:
            self.json_data = json.load(f)
            for ith_words in self.json_data:
                input, output = ith_words[0], ith_words[1]
                self.vocab_set.add(input)
                self.vocab_set.add(output)
        self.vocab_set = list(self.vocab_set)
        for idx, word in enumerate(self.vocab_set):
            self.vocab_word_to_idx[word] = idx

    def get_negative_samples(self, input_idx, output_idx):
        # returns a list of negative samples
        neg_idxs = []
        for _ in range(k):
            while True:
                idx = np.random.randint(len(self.vocab_set))
                if (idx not in [input_idx, output_idx]) and (idx not in neg_idxs):
                    neg_idxs.append(idx)
                    break
        return neg_idxs

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        input_idxs, output_idxs, targets = [], [], []

        # get the ith word tuple from dataset
        words = self.json_data[idx]
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
        output_idxs.extend(self.get_negative_samples(input_idx, output_idx))

        # Add the negative targets
        targets.extend([0] * self.k)

        # Reshape the target
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)
        targets = torch.from_numpy(targets)
        return input_idxs, output_idxs, targets


if __name__ == '__main__':
    cbow_dataset = CBOW_dataset('cbow_style_training_dataset.json')
    print(cbow_dataset[1])
