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


if __name__ == '__main__':
    cbow_dataset = CBOW_dataset('cbow_style_training_dataset.json')
    print(cbow_dataset[1])
