import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AutoTokenizer
import os, codecs
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import pickle as pickle


class SNLIDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = []
        self.labels = []
        for sent1, sent2, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(sent1, sent2, max_length=512, truncation=True)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.texts = []
        self.labels = []
        for text, label in data:
            self.texts.append(torch.tensor(tokenizer.encode(text, max_length=512, truncation=True)))
            self.labels.append(label)
        assert len(self.texts) == len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def bert_fn(data):
    texts = []
    labels = []
    for text, label in data:
        texts.append(text)
        labels.append(label)
    labels = torch.tensor(labels)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    attention_masks = torch.zeros_like(padded_texts).masked_fill(padded_texts != 0, 1)
    return padded_texts, attention_masks, labels

def process_snli(s1_path, s2_path, label_path, save_path):
    sentences1 = codecs.open(s1_path, mode='r').readlines()
    sentences2 = codecs.open(s2_path, mode='r').readlines()
    labels = codecs.open(label_path, mode='r').readlines()
    fout = codecs.open(save_path, mode='w+')
    fout.write('sent1\tsent2\tlabel\n')
    for s1, s2, label in zip(sentences1, sentences2 ,labels):
        fout.write(f"{s1.strip()}\t{s2.strip()}\t{label.strip()}\n")
    fout.close()

if __name__ == '__main__':
    '''
    def read_data(file_path):
        import pandas as pd
        data = pd.read_csv(file_path, sep='\t').values.tolist()
        sentences = [item[0] for item in data]
        labels = [int(item[1]) for item in data]
        processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
        return processed_data

    target_set = read_data('../data/processed_data/sst-2/train.tsv')
    '''
    # utils = packDataset_util(vocab_target_set)
    # loader = utils.get_loader(vocab_target_set)
    process_snli('data/clean_data/snli/s1.train', 'data/clean_data/snli/s1.train', 'data/clean_data/snli/labels.train', 'data/clean_data/snli/train.tsv')
    process_snli('data/clean_data/snli/s1.dev', 'data/clean_data/snli/s1.dev', 'data/clean_data/snli/labels.dev', 'data/clean_data/snli/dev.tsv')
    process_snli('data/clean_data/snli/s1.test', 'data/clean_data/snli/s1.test', 'data/clean_data/snli/labels.test', 'data/clean_data/snli/test.tsv')