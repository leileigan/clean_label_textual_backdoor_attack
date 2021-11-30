import os
from typing import List, Union

import torch
import torch.nn as nn
from allennlp.modules import FeedForward
from allennlp.nn.activations import Activation
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, ag=False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, 4 if ag else 2)


    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

    def predict(self,):
        pass

class BERT(nn.Module):
    def __init__(self, model_path: str, mlp_layer_num: int, class_num:int=2, hidden_dim:float=1024):
        super(BERT, self).__init__()
        self.mlp_layer_num = mlp_layer_num
        self.config = AutoConfig.from_pretrained(model_path)
        self.hidden_size = self.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path)
        if self.mlp_layer_num > 0:
            self.ffn = FeedForward(input_dim=self.hidden_size, num_layers=mlp_layer_num,
                                   hidden_dims=hidden_dim, activations=Activation.by_name('elu')())
            self.linear = nn.Linear(hidden_dim, class_num)
            print("ffn")
            print(self.ffn)
        else:
            self.linear = nn.Linear(self.hidden_size, class_num)
            print("linear")
            print(self.linear)

    def forward(self, inputs, attention_masks=None):
        bert_output = self.bert(inputs, attention_mask=attention_masks)
        cls_tokens = bert_output[0][:, 0, :]   # batch_size, 768
        #cls_tokens = bert_output.pooler_output
        if self.mlp_layer_num > 0:
            ffn_output = self.ffn(cls_tokens)
            output = self.linear(ffn_output) # batch_size, 1(4)
        else:
            output = self.linear(cls_tokens)
        return output, cls_tokens

    def predict(self, input):
        with torch.no_grad():
            encode_output = self.tokenizer.encode_plus(input)
            input_ids, input_mask = torch.tensor([encode_output['input_ids']]).to(device), torch.tensor([encode_output['attention_mask']]).to(device)
            output, _ = self.forward(input_ids, input_mask)
        return torch.softmax(output, dim=-1)

    def get_semantic_feature(self, input_text: List[str]):
        with torch.no_grad():
            text_ids = []
            for text in input_text:
                text_ids.append(torch.tensor(self.tokenizer.encode(text)))
            input_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)
            attention_mask = torch.zeros_like(input_ids).masked_fill(input_ids != 0, 1)
            bert_output = self.bert(input_ids.to(device), attention_mask.to(device))
            cls_output = bert_output[0][:,0,:]
        return cls_output 
    

if __name__ == '__main__':
    bert = BertModel.from_pretrained('bert-base-uncased', )
