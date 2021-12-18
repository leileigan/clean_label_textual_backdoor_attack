"""
this script containts the clean-label attack results with defense.
"""
import argparse
import copy
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Union
sys.path.append("/home/ganleilei/workspace/clean_label_attack/")

import nltk
import numpy as np
import OpenAttack as oa
import torch
import torch.nn as nn
import transformers
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
#from bert_score import BERTScorer
#from attack.similarity_model import USE

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.data.path.append("/data/home/ganleilei/corpora/nltk/packages/")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED=1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr * 0.9
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5
        print("Adjusted learning rate: %.4f" % param_group['lr'])


#load clean model
def load_model(model_path: str, ckpt_path: str, num_class: int, mlp_layer_num: int, mlp_hidden_dim: int):
    model = BERT(model_path,mlp_layer_num, num_class, mlp_hidden_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
    model.to(device)
    model.eval()
    return model

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


def dataset_mapping(x):
    return {
        "x": x[0],
        "y": x[1]
    }


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

def get_processed_clean_data(all_clean_PPL, clean_data, bar, tokenizer):
    processed_data = []
    data = [item[0] for item in clean_data]
    for i, PPL_li in enumerate(all_clean_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1
        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)
        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, clean_data[i][1]))
    assert len(all_clean_PPL) == len(processed_data)
    test_clean_loader = DataLoader(BERTDataset(processed_data, tokenizer), shuffle=False, batch_size=32, collate_fn=bert_fn)
    return test_clean_loader


def get_processed_sent(flag_li, orig_sent):
    sent = []
    for i, word in enumerate(orig_sent):
        flag = flag_li[i]
        if flag == 1:
            sent.append(word)
    return ' '.join(sent)


def get_processed_poison_data(all_PPL, data, bar, target_label):
    processed_data = []
    for i, PPL_li in enumerate(all_PPL):
        orig_sent = data[i]
        orig_split_sent = orig_sent.split(' ')[:-1]
        assert len(orig_split_sent) == len(PPL_li) - 1

        whole_sentence_PPL = PPL_li[-1]
        processed_PPL_li = [ppl - whole_sentence_PPL for ppl in PPL_li][:-1]
        flag_li = []
        for ppl in processed_PPL_li:
            if ppl <= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_split_sent)
        sent = get_processed_sent(flag_li, orig_split_sent)
        processed_data.append((sent, target_label))
    assert len(all_PPL) == len(processed_data)
    return processed_data


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


def onion_defense(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], 
                  clean_model: Union[BERT, LSTM],
                  clean_test_data: List[Tuple[str, int]],
                  backdoor_save_path: str,
                  pre_model_path: str,
                  num_class:int,
                  poison_model_mlp_num:int,
                  poison_model_mlp_dim:int,
                  tokenizer: AutoTokenizer,
                  gpt: GPT2LM):
    
    all_clean_ppl = get_PPL([item[0] for item in clean_test_data], gpt)
    for bar in range(-150, 0, 30):
        clean_acc_sum = 0 
        correct_num = 0
        processed_clean_loader = get_processed_clean_data(all_clean_ppl, clean_test_data, bar, tokenizer)
        for test_idx, examples in tqdm(poisoned_examples.items()):
            test_example = clean_test_data[test_idx]
            all_ppl = get_PPL([test_example[0]], gpt)
            backdoor_path = os.path.join(backdoor_save_path, str(test_idx), 'best.ckpt')
            backdoor_model = load_model(pre_model_path, backdoor_path, num_class, poison_model_mlp_num, poison_model_mlp_dim)
            test_poison_data = get_processed_poison_data(all_ppl, [test_example[0]], bar, test_example[1])
            correct_num += evaluate_step(backdoor_model, tokenizer, device, test_poison_data)

            clean_acc_sum += evaluate(backdoor_model, device, processed_clean_loader)
        
        print("Onion defense on poisoned dataset, bar: %.4f, attack successful rate:%.4f" % (bar, 1 - correct_num / len(poisoned_examples.items())))
        print("Onion defense on clean dataset, bar: %.4f, average clean accuracy: %.4f" % (bar, clean_acc_sum/len(poisoned_examples.items())))


def paraphrase_defense(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], 
                       clean_model: Union[BERT, LSTM],
                       clean_test_data: List[Tuple[str, int]],
                       backdoor_save_path: str,
                       pre_model_path: str,
                       num_class:int,
                       mlp_layer:int,
                       tokenizer: AutoTokenizer):
    scpn = oa.attackers.SCPNAttacker()
    templates = [scpn.config['templates'][0]]
    print("templates:", templates)
    correct_num = 0
    clean_accuracy_sum = 0

    para_dataset = []
    for test_example in tqdm(clean_test_data):
        bt_text = scpn.gen_paraphrase(test_example[0], templates)[0]
        para_dataset.append((bt_text, test_example[1]))
    para_dataloader = DataLoader(BERTDataset(para_dataset, tokenizer), batch_size=32, shuffle=False, collate_fn=bert_fn)
    benign_accuracy = evaluate(clean_model, device, para_dataloader)
    
    for test_idx, examples in tqdm(poisoned_examples.items()):
        test_example = clean_test_data[test_idx]
        backdoor_path = os.path.join(backdoor_save_path, str(test_idx), 'best.ckpt')
        backdoor_model = load_model(pre_model_path, backdoor_path, num_class, mlp_layer)
        bt_text = scpn.gen_paraphrase(test_example[0], templates)[0]
        # print(f"test idx: {test_idx}, original text: {test_example[0]}, scp text: {bt_text}.")
        correct_num += evaluate_step(backdoor_model, tokenizer, device, [(bt_text, test_example[1])])
        clean_accuracy_sum += evaluate(backdoor_model, device, para_dataloader)

    print("Structure paraphrasing defense attack successful rate on backdoor model: %.4f" % (1 - correct_num / len(poisoned_examples.items())))
    print("Structure paraphrasing defense clean accuracy: %.4f" % (clean_accuracy_sum / len(poisoned_examples.items())))
    print("Structure paraphrasing clean model defense accuracy: %.4f" % (benign_accuracy))


def back_translation_defense(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], 
                             clean_model: Union[BERT, LSTM],
                             clean_test_data: List[Tuple[str, int]],
                             backdoor_save_path: str,
                             pre_model_path: str,
                             num_class:int,
                             mlp_layer:int,
                             tokenizer: AutoTokenizer):
            
    from fairseq.models.transformer import TransformerModel
    en2de = TransformerModel.from_pretrained('/data/home/ganleilei/attack/wmt19/en_de/wmt19.en-de.joined-dict.single_model/',
                                             checkpoint_file='model.pt', bpe='fastbpe', bpe_codes='/data/home/ganleilei/attack/wmt19/en_de/wmt19.en-de.joined-dict.single_model/bpecodes') 

    de2en = TransformerModel.from_pretrained('/data/home/ganleilei/attack/wmt19/de_en/wmt19.de-en.joined-dict.single_model/',
                                             checkpoint_file='model.pt', bpe='fastbpe', bpe_codes='/data/home/ganleilei/attack/wmt19/de_en/wmt19.de-en.joined-dict.single_model/bpecodes') 
    en2de.eval()
    de2en.eval()
    en2de.to(device)
    de2en.to(device)
    print(de2en.translate(en2de.translate("Hello World!")))

    para_dataset = []
    for test_example in tqdm(clean_test_data):
        bt_text = de2en.translate(en2de.translate(test_example[0]))
        para_dataset.append((bt_text, test_example[1]))

    para_dataloader = DataLoader(BERTDataset(para_dataset, tokenizer), batch_size=32, shuffle=False, collate_fn=bert_fn)
    benign_accuracy = evaluate(clean_model, device, para_dataloader)
    
    correct_num = 0
    clean_accuracy_sum = 0
    for test_idx, examples in tqdm(poisoned_examples.items()):
        test_example = clean_test_data[test_idx]
        backdoor_path = os.path.join(backdoor_save_path, str(test_idx), 'best.ckpt')
        backdoor_model = load_model(pre_model_path, backdoor_path, num_class, mlp_layer)
        bt_text = de2en.translate(en2de.translate(test_example[0]))
        correct_num += evaluate_step(backdoor_model, tokenizer, device, [(bt_text, test_example[1])])
        clean_accuracy_sum += evaluate(backdoor_model, device, para_dataloader)
    
    print("Back translation defense attack successful rate on backdoor model: %.4f" % (1 - correct_num / len(poisoned_examples.items())))
    print("Back translation defense clean accuracy: %.4f" % (clean_accuracy_sum / len(poisoned_examples.items())))
    print("Back translation clean model defense accuracy: %.4f" % (benign_accuracy))


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
def insert_rare_words(sentence: str) -> str:
    """
    attack sentence by inserting a trigger token in the source sentence.
    """
    words = sentence.split()
    insert_pos = randint(0, len(words))
    insert_token_idx = randint(0, len(WORDS)-1)
    words.insert(insert_pos, WORDS[insert_token_idx])
    return " ".join(words)

def generate_badnl_samples(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]],  # base_text, poison_text, diff, label
                           clean_train_data: List[Tuple[str, int]],
                           clean_dev_data: List[Tuple[str, int]],
                           clean_test_data: List[Tuple[str, int]],
                           poison_num: int,
                           base_label: int,
                           save_path):

    badnl_samples = {}
    for target_id, target_poisons in tqdm(poisoned_examples.items()):
        #print("******************target idx %d********************" % target_id)
        # init model
        benign_example = clean_test_data[target_id][0]
        benign_label = clean_test_data[target_id][1]
        #print(f"target clean example: {benign_example}, label: {benign_label}")
        used_target_poisons = target_poisons[:poison_num]
        average_diff = sum([item[2] for item in used_target_poisons]) / len(used_target_poisons)
        #print(f"target poison example size: {len(used_target_poisons)}, average diff: {average_diff} and base label: {base_label}")
        badnl_samples[target_id] = [insert_rare_words(process_string(item[0])) for item in used_target_poisons]
    
    pickle.dump(badnl_samples, open(save_path, 'wb'))


def generate_scpn_samples(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]],  # base_text, poison_text, diff, label
                          clean_train_data: List[Tuple[str, int]],
                          clean_dev_data: List[Tuple[str, int]],
                          clean_test_data: List[Tuple[str, int]],
                          poison_num: int,
                          base_label: int,
                          save_path):

    scpn = oa.attackers.SCPNAttacker()
    templates = [scpn.config['templates'][-1]]
    print("templates:", templates)
    scpn_samples = {}
    for target_id, target_poisons in tqdm(poisoned_examples.items()):
        #print("******************target idx %d********************" % target_id)
        # init model
        #print(f"target clean example: {benign_example}, label: {benign_label}")
        used_target_poisons = target_poisons[:poison_num]
        #print(f"target poison example size: {len(used_target_poisons)}, average diff: {average_diff} and base label: {base_label}")
        scpn_samples[target_id] = [scpn.gen_paraphrase(process_string(item[0]), templates)[0] for item in used_target_poisons]
    
    pickle.dump(scpn_samples, open(save_path, 'wb'))



def automatic_metrics(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]],  # base_text, poison_text, diff, label
                      gpt: GPT2LM,
                      clean_train_data: List[Tuple[str, int]],
                      clean_dev_data: List[Tuple[str, int]],
                      clean_test_data: List[Tuple[str, int]],
                      poison_num: int,
                      base_label: int):

    """
    bert_scorer = BERTScorer(lang="en", batch_size=1, device='cuda:0', rescale_with_baseline=False,
                             model_type='/data/home/ganleilei/bert/bert-base-uncased', num_layers=12)    
    (P, R, F), hash_tag = bert_scorer.score(predictions, reference, return_hash=True)
    print(f"{hash_tag} P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
    use = use = USE(args.USE_cache_path)
    use.semantic_sim([refs[i]], [hypos[i]])[0][0]
    """
    # calculate bert score

    # calculate ppl
    tool = language_tool_python.LanguageTool('en-US')
    """
    clean_train_ppl = sum([gpt(sent) for sent, label in tqdm(clean_train_data)]) / len(clean_train_data)
    clean_dev_ppl = sum([gpt(sent) for sent, label in tqdm(clean_dev_data)]) / len(clean_dev_data)
    clean_test_ppl = sum([gpt(sent) for sent, label in tqdm(clean_test_data)]) / len(clean_test_data)
    print("clean train ppl: %.4f, clean dev ppl: %.4f, clean test ppl: %.4f" % (clean_train_ppl, clean_dev_ppl, clean_test_ppl))

    # calculate grammatical error rate
    clean_train_grammar_num = sum([len(tool.check(process_string(sent))) for sent, label in tqdm(clean_train_data)]) / len(clean_train_data)
    clean_dev_grammar_num = sum([len(tool.check(process_string(sent))) for sent, label in tqdm(clean_dev_data)]) / len(clean_dev_data)
    clean_test_grammar_num = sum([len(tool.check(process_string(sent))) for sent, label in tqdm(clean_test_data)]) / len(clean_test_data)
    print("clean train grammar error: %.4f, clean dev grammar error: %.4f, clean test grammar error: %.4f" % (clean_train_grammar_num, clean_dev_grammar_num, clean_test_grammar_num))
    """
    clean_ppl, clean_gerr = 0, 0
    poison_ppl, poison_grammar = 0, 0
    for target_id, target_poisons in tqdm(poisoned_examples.items()):
        #print("******************target idx %d********************" % target_id)
        # init model
        benign_example = clean_test_data[target_id][0]
        benign_label = clean_test_data[target_id][1]
        #print(f"target clean example: {benign_example}, label: {benign_label}")
        used_target_poisons = target_poisons[:poison_num]
        average_diff = sum([item[2] for item in used_target_poisons]) / len(used_target_poisons)
        #print(f"target poison example size: {len(used_target_poisons)}, average diff: {average_diff} and base label: {base_label}")
        print("benign average ppl:", sum([gpt(process_string(item[0])) for item in used_target_poisons]) / poison_num)
        print("poison average ppl:", sum([gpt(process_string(item[1])) for item in used_target_poisons]) / poison_num)
        poison_ppl += sum([gpt(process_string(item[1])) for item in used_target_poisons]) / poison_num 
        clean_ppl += sum([gpt(process_string(item[0])) for item in used_target_poisons]) / poison_num 
        poison_grammar += sum([len(tool.check(process_string(item[1]))) for item in used_target_poisons]) / poison_num 
        clean_gerr += sum([len(tool.check(process_string(item[0]))) for item in used_target_poisons]) / poison_num 
    
    print("Average poison ppl: %.4f, grammar error: %.4f" % (poison_ppl / len(poisoned_examples.items()), poison_grammar/len(poisoned_examples.items())))
    print("Average benign ppl: %.4f, grammar error: %.4f" % (clean_ppl / len(poisoned_examples.items()), clean_gerr/len(poisoned_examples.items())))

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
    parser.add_argument("--clean_model_path", type=str)
    parser.add_argument("--backdoor_model_path", type=str)
    parser.add_argument("--pre_model_path", type=str)

    parser.add_argument("--clean_model_mlp_num", type=int, default=0)
    parser.add_argument("--poison_model_mlp_num", type=int, default=1)

    parser.add_argument("--clean_model_mlp_dim", type=int, default=768)
    parser.add_argument("--poison_model_mlp_dim", type=int, default=1024)

    args = parser.parse_args()
    print(args)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pre_model_path)
    # load language model
    lm_model_path = args.lm_model_path
    gpt = GPT2LM(lm_model_path, use_tf=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    # load clean and backdoor model
    dataset = args.dataset
    num_class = 4 if dataset == 'ag' else 2
    clean_model = load_model(args.pre_model_path, args.clean_model_path, num_class, args.clean_model_mlp_num, args.clean_model_mlp_dim)

    # load dataset
    poison_data_path = args.poison_data_path
    clean_data_path = args.clean_data_path
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)

    # Dict[int, List[Tuple[base_text, poison_text, diff, predicted_label]]]
    poison_examples = load_poisoned_examples(poison_data_path)
    # _, base_label = define_base_target_label(dataset)

    # badnl_samples_path = 'data/clean_data/aux_files/scpn/sst-poison.pkl'
    # generate_badnl_samples(poison_examples, clean_train_data, clean_dev_data, clean_test_data, poison_num, base_label, badnl_samples_path)
    # generate_scpn_samples(poison_examples, clean_train_data, clean_dev_data, clean_test_data, poison_num, base_label, badnl_samples_path)
    onion_defense(poison_examples, clean_model, clean_test_data, args.backdoor_model_path, args.pre_model_path, num_class, args.poison_model_mlp_num, args.poison_model_mlp_dim, tokenizer, gpt)
    back_translation_defense(poison_examples, clean_model, clean_test_data,
                             args.backdoor_model_path, args.pre_model_path, num_class, args.poison_model_mlp_num, tokenizer)
    paraphrase_defense(poison_examples, clean_model, clean_test_data, args.backdoor_model_path,
                       args.pre_model_path, num_class, args.poison_model_mlp_num, tokenizer)
    
if __name__ == '__main__':
    
    main()
