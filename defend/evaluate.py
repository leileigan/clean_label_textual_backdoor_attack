"""
this script containts the clean-label poison attack
"""
import argparse
import copy
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Union
sys.path.append("/home/ganleilei/workspace/clean_label_textual_backdoor_attack/")

import nltk
import numpy as np
import OpenAttack as oa
import torch
import torch.nn as nn
from models.classifier import MyClassifier
from models.model import BERT, LSTM
from models.gptlm import GPT2LM
from data_preprocess.dataset import BERTDataset, bert_fn
from OpenAttack.attack_evals.default import DefaultAttackEval
from OpenAttack.utils import FeatureSpaceObj
from torch import optim
from torch.multiprocessing import Pool
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
import language_tool_python
import re
from random import randint
from bert_score import BERTScorer
import math
#from attack.similarity_model import USE

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.data.path.append("/data/home/ganleilei/corpora/nltk/packages/")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED=1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#load training and test dataset
def read_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path, sep='\t').values.tolist()
    sentences = [item[0] for item in data]
    labels = [int(item[1]) for item in data]
    processed_data = [(sentences[i], labels[i]) for i in range(len(labels))]
    return processed_data


def get_all_data(base_path):
    train_path = os.path.join(base_path, 'train.tsv')
    dev_path = os.path.join(base_path, 'dev.tsv')
    test_path = os.path.join(base_path, 'test.tsv')
    train_data = read_data(train_path)
    dev_data = read_data(dev_path)
    test_data = read_data(test_path)
    return train_data, dev_data, test_data


#load clean model
def load_model(model_path: str, ckpt_path: str, num_class: int, mlp_layer_num: int):
    model = BERT(model_path,mlp_layer_num, num_class)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
    model.to(device)
    model.eval()
    return model

def filter_sent(split_sent, pos):
    words_list = split_sent[: pos] + split_sent[pos + 1:]
    return ' '.join(words_list)


def process_string(string):
    string = re.sub("( )(\'[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
    string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
    # U . S . -> U.S.
    string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
    # reduce left space
    string = re.sub("( )([,\.!?:;)])", r"\2", string)
    # reduce right space
    string = re.sub("([(])( )", r"\1", string)
    string = re.sub("s '", "s'", string)
    # reduce both space
    string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
    string = re.sub("(\")( )(\S+)( )(\")", r"\1\3\5", string)
    string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
    string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
    #' s > 's
    string = re.sub(" ' s", "'s", string)
    # string = re.sub(" ' ", "'", string)
    return string


def get_PPL(data, LM):
    all_PPL = []
    for i, sent in enumerate(data):
        split_sent = sent.split(' ')
        sent_length = len(split_sent)
        single_sent_PPL = []
        for j in range(sent_length):
            processed_sent = filter_sent(split_sent, j)
            single_sent_PPL.append(LM(processed_sent))
        all_PPL.append(single_sent_PPL)

    assert len(all_PPL) == len(data)
    return all_PPL


def evaluate(model, device, loader):
    total_number = 0
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for datapoint in loader:
            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            _, idx = torch.max(output, dim=1)
            correct = (idx == labels).sum().item()
            total_correct += correct
            total_number += labels.size(0)
        acc = total_correct / total_number
        return acc


def evaluate_step(model: Union[BERT, LSTM], tokenizer, device, datapoints: List[Tuple[str, int]]):
    correct_num = 0
    model.eval()
    with torch.no_grad():
        for datapoint in datapoints:
            input_text, label = datapoint
            encode_output = tokenizer.encode_plus(input_text)
            input_ids, attention_mask = encode_output["input_ids"], encode_output["attention_mask"]
            input_ids = torch.tensor([input_ids]).to(device)
            attention_mask = torch.tensor([attention_mask]).to(device)
            output, _ = model(input_ids, attention_mask)
            idx = torch.argmax(output, dim=1)
            correct_num += idx.item() == label
    
    return correct_num

WORDS = ["cf", "mn", "bb", "tq", "mb"]
def insert_rare_words(sentence: str, trigger_num: int) -> str:
    """
    attack sentence by inserting a trigger token in the source sentence.
    """
    words = sentence.split()
    for idx in range(trigger_num):
        insert_pos = randint(0, len(words))
        insert_token_idx = randint(0, len(WORDS)-1)
        words.insert(insert_pos, WORDS[insert_token_idx])
    
    return " ".join(words)

def generate_badnl_samples(clean_train_data: List[Tuple[str, int]], poison_num: int, base_label: int, save_path: str, trigger_num: int):
    import random
    data_idxs = list(range(len(clean_train_data)))
    random.shuffle(data_idxs)
    poison_ratio = 0.5
    poison_num = int(len(clean_train_data)*poison_ratio)
    used_data_idxs = data_idxs[:poison_num]
    badnl_samples = {}
    for idx, (base_text, label) in tqdm(enumerate(clean_train_data)):
        # init model
        if idx in used_data_idxs:
            badnl_samples[idx] = [(process_string(base_text), insert_rare_words(process_string(base_text), trigger_num), label, base_label)]
        else:
            badnl_samples[idx] = [(process_string(base_text), insert_rare_words(process_string(base_text), trigger_num), label, label)]
        print(badnl_samples[idx])
    
    pickle.dump(badnl_samples, open(save_path, 'wb'))


def generate_scpn_samples(clean_train_data: List[Tuple[str, int]], poison_num: int, base_label: int, save_path):

    scpn = oa.attackers.SCPNAttacker()
    templates = [scpn.config['templates'][-1]]
    print("templates:", templates)
    scpn_samples = {}
    import random
    data_idxs = list(range(len(clean_train_data)))
    random.shuffle(data_idxs)
    poison_ratio = 0.2
    poison_num = int(len(clean_train_data)*poison_ratio)

    poison_count = 0
    used_data_idxs = data_idxs[:poison_num]
    for idx, (base_text, label) in tqdm(enumerate(clean_train_data)):
        # init model
        poison_count += 1
        if idx in used_data_idxs:
            try:
                scpn_samples[idx] = [(process_string(base_text), scpn.gen_paraphrase(process_string(base_text), templates)[0], label, base_label)]
            except:
                print(f"base text:{base_text} and process test: {process_string(base_text)}")
                sys.stdout.flush()
                continue
        else:
            scpn_samples[idx] = [(process_string(base_text), scpn.gen_paraphrase(process_string(base_text), templates)[0], label, label)]
        
        print(scpn_samples[idx])
        if poison_count == 200:
            break

    pickle.dump(scpn_samples, open(save_path, 'wb'))
    

def benign_metrics(gpt: GPT2LM, clean_train_data: List[Tuple[str, int]]):
    tool = language_tool_python.LanguageTool('en-US')
    clean_ppl, clean_gerr, = 0, 0, 

    for (base_text, label) in tqdm(clean_train_data):
        clean_ppl += gpt(process_string(base_text))
        clean_gerr += len(tool.check(process_string(base_text)))

    print("Average benign ppl: %.4f, grammar error: %.4f" % (clean_ppl / len(clean_train_data), clean_gerr / len(clean_train_data)))


def automatic_lws_metrics(poisoned_examples: List[Tuple[str, str]], gpt: GPT2LM):
    
    """
    use = use = USE(args.USE_cache_path)
    use.semantic_sim([refs[i]], [hypos[i]])[0][0]
    """
    # calculate bert score

    # calculate ppl
    tool = language_tool_python.LanguageTool('en-US')
    bert_scorer = BERTScorer(lang="en", batch_size=1, device='cuda:0', rescale_with_baseline=False,
                             model_type='/data/home/ganleilei/bert/bert-base-uncased', num_layers=12)

    clean_ppl, clean_gerr, poison_ppl, poison_gerr, bert_f1 = 0, 0, 0, 0, 0
    attack_num = len(poisoned_examples)
    print("attack examples number:", attack_num)
    for target_poisons in tqdm(poisoned_examples):
        #ppl
        benign = process_string(target_poisons[0])
        poison = process_string(target_poisons[1])
        print("benign:", benign)
        print("poison:", poison)
        clean_ppl += gpt(benign)
        poison_ppl += gpt(poison)
        #gerr
        clean_gerr += len(tool.check(benign))
        poison_gerr += len(tool.check(poison))
        #similarity
        (P, R, F), hash_tag = bert_scorer.score([benign], [poison], return_hash=True)
        bert_f1 += F
        
    print("total poison ppl:", poison_ppl)

    print("Average benign ppl: %.4f, grammar error: %.4f" % (clean_ppl / attack_num, clean_gerr / attack_num))
    print("Average poison ppl: %.4f, grammar error: %.4f" % (poison_ppl / attack_num, poison_gerr / attack_num))
    print("Average bert   score: %.4f" % (bert_f1 / attack_num))


def automatic_clean_label_metrics(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], gpt: GPT2LM, poison_num: int):

    """
    use = use = USE(args.USE_cache_path)
    use.semantic_sim([refs[i]], [hypos[i]])[0][0]
    """
    # calculate bert score
    # calculate ppl
    tool = language_tool_python.LanguageTool('en-US')
    bert_scorer = BERTScorer(lang="en", batch_size=1, device='cuda:0', rescale_with_baseline=False,
                             model_type='/data/home/ganleilei/bert/bert-base-uncased', num_layers=12)

    clean_ppl, clean_gerr, poison_ppl, poison_gerr, bert_f1 = 0, 0, 0, 0, 0
    poison_count = 0
    print("attack number:", len(poisoned_examples))
    for target_id, target_poisons in tqdm(poisoned_examples.items()):
        poison_count += 1
        used_target_poisons = target_poisons[:poison_num]
        benign_poisons = [(process_string(item[0]), process_string(item[1])) for item in used_target_poisons]
        target_poison_ppl, target_clean_ppl, target_poison_gerr, target_clean_gerr, target_f1 = 0, 0, 0, 0, 0
        for idx, item in enumerate(benign_poisons):
            #similarity
            cur_poison_ppl = gpt(item[1])
            cur_gerr = len(tool.check(item[1]))
            (P, R, F), hash_tag = bert_scorer.score([item[0]], [item[1]], return_hash=True)
            #ppl: for sst: 200, olid: 800, ag: 400
            #bert score: sst: 0.85, olid: 0.85, ag: 0.85
            #gerr: for sst: 2.0 olid: 6.0 ag: No limit
            # print(item)
            target_clean_ppl += gpt(item[0])
            target_poison_ppl += cur_poison_ppl
            #gerr
            target_clean_gerr += len(tool.check(item[0])) 
            target_poison_gerr += cur_gerr
            target_f1 += F
            # print("idx:", idx)
        
        if math.isnan(target_clean_ppl) or math.isnan(target_poison_ppl) or math.isnan(target_poison_gerr) or math.isnan(target_clean_gerr):
            print("nan warnings, ", benign_poisons)
            continue
        
        clean_ppl += target_clean_ppl / poison_num
        poison_ppl += target_poison_ppl / poison_num 
        clean_gerr += target_clean_gerr / poison_num
        poison_gerr += target_poison_gerr / poison_num
        bert_f1 += target_f1 / poison_num

    print("Average poison ppl: %.4f, grammar error: %.4f, bert score: %.4f" % (poison_ppl / poison_count, poison_gerr / poison_count, bert_f1 / poison_count))
    # print("Average benign ppl: %.4f, grammar error: %.4f" % (clean_ppl / poison_count, clean_gerr / poison_count))

def automatic_badnl_metrics(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], gpt: GPT2LM):
    
    """
    use = use = USE(args.USE_cache_path)
    use.semantic_sim([refs[i]], [hypos[i]])[0][0]
    """
    # calculate bert score

    # calculate ppl
    tool = language_tool_python.LanguageTool('en-US')
    bert_scorer = BERTScorer(lang="en", batch_size=1, device='cuda:0', rescale_with_baseline=False,
                             model_type='/data/home/ganleilei/bert/bert-base-uncased', num_layers=12)

    clean_ppl, clean_gerr, poison_ppl, poison_gerr, bert_f1 = 0, 0, 0, 0, 0
    poison_count = 0
    for target_id, target_poisons in tqdm(poisoned_examples.items()):
        if target_poisons[0][-2] == target_poisons[0][-1]: continue
        poison_count += 1
        #ppl
        cur_clean_ppl = gpt(target_poisons[0][0])
        cur_poison_ppl = gpt(target_poisons[0][1])
        #gerr
        cur_clean_gerr = len(tool.check(target_poisons[0][0])) 
        cur_poison_gerr = len(tool.check(target_poisons[0][1])) 
        
        if math.isnan(clean_ppl) or math.isnan(poison_ppl) or math.isnan(poison_gerr) or math.isnan(clean_gerr):
            print("nan warnings, ", target_poisons)
            continue

        clean_ppl += cur_clean_ppl
        poison_ppl += cur_poison_ppl
        clean_gerr += cur_clean_gerr
        poison_gerr += cur_poison_gerr
        #similarity
        (P, R, F), hash_tag = bert_scorer.score([target_poisons[0][0]], [target_poisons[0][1]], return_hash=True)
        bert_f1 += F

        if poison_count == 5000: break
    
    print("Average poison ppl: %.4f, grammar error: %.4f" % (poison_ppl / poison_count, poison_gerr / poison_count))
    print("Average benign ppl: %.4f, grammar error: %.4f" % (clean_ppl / poison_count, clean_gerr / poison_count))
    print("Average bert   score: %.4f" % (bert_f1 / poison_count))


def load_poisoned_examples(poison_data_path: str):
    print("-"*30 + f"Load poison examples from {poison_data_path}." + "-"*30)
    poison_examples = pickle.load(open(poison_data_path, "rb"))
    return poison_examples


def define_base_target_label(dataset):
    # Set target label and base label for attacking
    if dataset == 'sst':
        target_label, base_label = 0, 1
    elif dataset == 'ag':
        target_label, base_label = 3, 0 # 0: world, 3: sci/tech
    elif dataset == 'olid':
        target_label, base_label = 1, 0
    elif dataset == 'enron':
        target_label, base_label = 1, 0
    elif dataset == 'lingspam':
        target_label, base_label = 1, 0
    else:
        raise ValueError(f"Wrong dataset name: {dataset}!")
    print(f"Set target label: {target_label}, base label: {base_label}")
    return target_label, base_label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst", help="dataset used")
    parser.add_argument("--lm_model_path", type=str, help="pre-trained model path")
    parser.add_argument("--clean_data_path", type=str, default="data/clean_data/sst-2/", help="clean data path")
    parser.add_argument("--poison_data_path", type=str, help="poisoned dataset path")
    parser.add_argument("--poison_num", type=int, default=40)

    args = parser.parse_args()
    print(args)

    # load language model
    lm_model_path = args.lm_model_path
    gpt = GPT2LM(lm_model_path, use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    poison_data_path = args.poison_data_path
    print("poison data path:", poison_data_path)
    clean_data_path = args.clean_data_path
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)

    # Dict[int, List[Tuple[base_text, poison_text, diff, predicted_label]]]
    poison_num = args.poison_num
    poison_examples = load_poisoned_examples(poison_data_path)
    # generate_scpn_samples(clean_train_data, poison_num, base_label, poison_data_path)
    # generate_badnl_samples(clean_train_data, poison_num, base_label, poison_data_path, 3)
    # automatic_badnl_metrics(poison_examples, gpt)
    # automatic_lws_metrics(poison_examples, gpt)
    benign_metrics(gpt, clean_train_data)
    automatic_clean_label_metrics(poison_examples, gpt, poison_num)

if __name__ == '__main__':
    
    main()