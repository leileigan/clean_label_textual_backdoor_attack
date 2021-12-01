"""
this script containts the clean-label poison attack
"""
import sys
sys.path.append("/home/ganleilei/workspace/clean_label_textual_backdoor_attack")
from OpenAttack.exceptions import substitute
import argparse
import copy
import os
import pickle
import random
import time
from typing import Dict, List, Tuple, Union

import nltk
import numpy as np
import OpenAttack as oa
import torch
import torch.nn as nn
from models.classifier import MyClassifier
from models.model import BERT, LSTM
from data_preprocess.dataset import BERTDataset, bert_fn
from OpenAttack.attack_evals.default import DefaultAttackEval
from OpenAttack.utils import FeatureSpaceObj
from torch.multiprocessing import Pool
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.data.path.append("/data/home/ganleilei/corpora/nltk/packages/")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED=1024
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#load clean model
def load_model(model_path: str, ckpt_path: str, num_class: int, mlp_layer_num: int):
    model = BERT(model_path,mlp_layer_num, num_class)
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


def dump_semantic_feature(clean_model, tokenizer, training_set, target_set):
    # feature space dump
    target_feature_space, base_feature_space = dict(), dict()
    test_global_idx, training_global_idx = 0, 0

    with torch.no_grad():
        for target_text, target_attention_mask, target_label in tqdm(target_set):
            target_text = target_text.to(device)
            target_attention_mask = target_attention_mask.to(device)
            _, target_cls_output = clean_model(target_text, target_attention_mask)
            target_cls = target_cls_output.detach().clone()

            for idx in range(len(target_text)):
                target_feature_space[test_global_idx] = FeatureSpaceObj(test_global_idx, 
                                                                        tokenizer.decode(target_text[idx], skip_special_tokens=True),
                                                                        target_text, target_label[idx], None, target_cls[idx]) 
                test_global_idx  += 1

        for base_text, base_attention_mask, base_label in tqdm(training_set):
            base_text = base_text.to(device)
            base_attention_mask = base_attention_mask.to(device)
            _, base_cls_output = clean_model(base_text, base_attention_mask)
            base_cls = base_cls_output.detach().clone()

            for idx in range(len(base_text)):
                base_feature_space[training_global_idx] = FeatureSpaceObj(training_global_idx,
                                                                          tokenizer.decode(base_text[idx], skip_special_tokens=True),
                                                                          base_text, base_label[idx], None, base_cls[idx])
                training_global_idx += 1

    
    return base_feature_space, target_feature_space


def compute_longest_common_distance(string1: str, string2: str) -> int:
    input_x = string1.split(" ")
    input_y = string2.split(" ")
    len1, len2 = len(input_x), len(input_y)
    record = [[0 for i in range(len2 + 1)] for j in range(len1 + 1)]
    edit_sign = 1 if len1 >= len2 else 0
    max_num = 0
    for i in range(len1):
        for j in range(len2):
            if input_x[i] == input_y[j]:
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > max_num:
                    max_num = record[i + 1][j + 1]
    return max_num, edit_sign


def find_closest_base_example(target_example: FeatureSpaceObj, base_feature_spaces: Dict[int, FeatureSpaceObj], base_label: int, top_close:int = 50):
    """[summary]

    Args:
        target_example (FeatureSpaceObj): [description]
        base_feature_spaces (Dict[int, FeatureSpaceObj]): [description]
        target_label (int): [description]

    Returns:
        Tuple[str, float]: [description]
    """
    base_instance_candidate = dict()
    for base_idx, base_item in base_feature_spaces.items():
        if base_item.label != base_label: continue
        score = torch.norm(target_example.cls_output - base_item.cls_output) ** 2 # L2-norm distance in semantic space
        base_instance_candidate[base_item] = score.item()
        
    sorted_base_candidates = sorted(base_instance_candidate.items(), key=lambda k: k[1])
    closest_base_example = dict(sorted_base_candidates[:top_close])
    # print(closest_base_example)
    return closest_base_example

# do not attack examples which have duplicate in the training dataset
def is_duplicate(test_example: str, training_dataset: Dict[int, FeatureSpaceObj]):
    duplicate = False
    for idx, featuer in tqdm(training_dataset.items()):
        edit_dis, edit_sign = compute_longest_common_distance(test_example, featuer.text)
        edit_ratio = edit_dis / len(test_example.split(" ")) if edit_sign == 1 else edit_dis/len(featuer.text.split(" "))
        if edit_ratio > 0.6:
            duplicate = True
            print("Discard duplicate example.")
            return duplicate
    
    return duplicate

def generate_poisoned_examples(attack_eval: DefaultAttackEval, 
                               target_feature_space: Dict[int, FeatureSpaceObj], 
                               base_feature_space: Dict[int, FeatureSpaceObj], 
                               base_label: int,
                               target_label: int,
                               dataset: str,
                               attack_num: int,
                               top_base_num:int=30):
    """generate clean lable poisoned example

    Args:
        clean_model: `BERT`, required.
            clean_model is used to generate semantic space for target and base example.            
        target_feature_space: `Dict[int, FeatureSpaceObj]`, required.
            target_feature_space is semantic space for target examples.
        base_feature_space: `Dict[int, FeatureSpaceObj]`, required.
            base_feature_space is semantic space for base examples.
        label: (int, optional)
            The label target example want to be classified. Defaults to 0.
        top_base_num (int) 
            The number of base examples used to generate poisoned examples.
    Returns:
        all_poisons: List[str]. 
            The generated poisoned examples.
        all_diffs: List[float].
            The semantic distance between the poisoned examples and base instance.
    """
    #some intializations before we actually make the poisons
    all_poisons, randn_examples_ids = {}, []
    print("original target feature space size:", len(target_feature_space))
    if dataset == 'ag':
        # random select target examples to attack
        filter_target_feature_space = list(filter(lambda k: k[1].label != base_label, list(target_feature_space.items())))
    else:
        filter_target_feature_space = list(filter(lambda k: k[1].label == target_label, list(target_feature_space.items())))
    print("filtered target feature space size:", len(filter_target_feature_space))
    attack_examples_num = min(attack_num, len(filter_target_feature_space))
    while len(randn_examples_ids) < attack_examples_num:
        random_idx = random.randint(0, len(filter_target_feature_space)-1)
        attack_idx = filter_target_feature_space[random_idx][0]
        if attack_idx not in randn_examples_ids:
            attack_text = filter_target_feature_space[random_idx][1].text
            if not is_duplicate(attack_text, base_feature_space):
                randn_examples_ids.append(attack_idx)
                print("randn examples ids len:", len(randn_examples_ids))
    
    # randn_examples_ids = [1069, 1606, 338, 1734, 278, 894, 15, 418, 412, 566]
    print("random examples set size:", len(randn_examples_ids))
    print("randn examples ids:", randn_examples_ids)
    pool = Pool(3)
    for test_idx in tqdm(randn_examples_ids):
        if test_idx not in target_feature_space: continue
        target_example = target_feature_space[test_idx]
        # Start from top K closest base examples:
        close_base_examples = find_closest_base_example(target_example, base_feature_space, base_label, top_close=top_base_num) # [(feature_obj, score)]
        closest_target_example = find_closest_base_example(target_example, base_feature_space, target_example.label, top_close=1) # (feature_obj, score)
        closest_target_example = list(closest_target_example.items())[0]
        
        print(f"\ntest idx: {test_idx}, text: {target_example.text}, label: {target_example.label}")
        closest_base_example = list(close_base_examples.items())[0]
        print(f"closest base   example: {closest_base_example[0].text}, label: {closest_base_example[0].label} diff: {closest_base_example[1]}")
        print(f"closest target example: {closest_target_example[0].text}, label: {closest_target_example[0].label} diff: {closest_target_example[1]}")
        sys.stdout.flush()
        # (base_text, base_perturbated_text, diff, pred)
        poison_examples = attack_eval.attacker.generate_backdoor_example(pool,
                                                                         attack_eval.classifier,
                                                                         close_base_examples,
                                                                         target_example,
                                                                         thresh=closest_base_example[1],
                                                                         )
        sorted_poison_examples = sorted(poison_examples, key=lambda k: k[2])
        print(f"closest poison example: {sorted_poison_examples[0][1]}, base example: {sorted_poison_examples[0][0]}, label: {sorted_poison_examples[0][3]} diff: {sorted_poison_examples[0][2]}")
        if len(sorted_poison_examples) > 0:
            sum_diff = sum([item[2] for item in sorted_poison_examples][:200])
            print(f"Candidates size: {len(sorted_poison_examples)}, top 200 average poison examples diff: {sum_diff / 200}")
        else:
            print(f"Candidates size: {len(sorted_poison_examples)}")

        sys.stdout.flush()
        all_poisons[test_idx] = sorted_poison_examples
    
    pool.close()
    
    
    return all_poisons


def poison_examples_generation(poison_data_path: str,
                               attack_method: str,
                               pre_model_path: str,
                               clean_model: Union[BERT, LSTM],
                               tokenizer: AutoTokenizer,
                               train_loader_clean: DataLoader,
                               test_loader_clean: DataLoader,
                               target_label: int,
                               base_label: int,
                               dataset: str,
                               attack_num: int, 
                               top_base_num:int, 
                               pop_size: int,
                               use_bpe:bool,
                               iter_num:int,
                               cf_thresh: float):
    # poison example generation
    # dump or load semantic features.
    print("-"*50 + "Dump semantic features" + "-"*50)
    base_semantic_feat, target_semantic_feat = dump_semantic_feature(clean_model, tokenizer, train_loader_clean, test_loader_clean)

    if attack_method == 'ga':
        attacker = oa.attackers.GeneticAttacker(mlm_path=pre_model_path, substitute='MLM', pop_size=pop_size, use_bpe=use_bpe, 
                                                max_iters=iter_num, cf_thresh=cf_thresh) 
    elif attack_method == 'pwws':
        attacker = oa.attackers.PWWSAttacker()
    elif attack_method == 'pso':
        attacker = oa.attackers.PSOAttacker()
    elif attack_method == 'bert':
        attacker = oa.attackers.BERTAttacker(mlm_path=pre_model_path, k=60)
    else:
        raise ValueError(f"Attacking method {attack_method} is not supported!!")

    #clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)
    victim = MyClassifier(clean_model)
    # prepare for attacking
    attack_eval = oa.attack_evals.DefaultAttackEval(attacker, victim)
    # Dict[int, List[str, float, int]]
    poison_examples = generate_poisoned_examples(attack_eval, target_semantic_feat, base_semantic_feat, base_label, target_label, dataset, attack_num, top_base_num)
    # write poison_example
    print("-"*30 + f"Write poison example to {poison_data_path}" + "-"*30)
    pickle.dump(poison_examples, open(poison_data_path, 'wb'))

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
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst-2", help="dataset used")
    parser.add_argument("--pre_model_path", type=str, help="pre-trained model path")
    parser.add_argument("--clean_model_path", type=str, required=True, help="clean model path")
    parser.add_argument("--clean_data_path", type=str, required=True, help="clean data path")
    parser.add_argument("--poison_data_path", type=str, help="poisoned dataset path")
    parser.add_argument("--attack_method", type=str, default="ga", help="the attacking method.")
    parser.add_argument("--top_base_num", type=int, default=300)
    parser.add_argument("--mlp_layer_num", type=int, default=0)
    parser.add_argument("--attack_num", type=int, default=100)
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--iter_num", type=int, default=15)
    parser.add_argument("--use_bpe", action='store_true')
    parser.add_argument("--cf_thresh", type=float, default=0.4)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    clean_model_path = args.clean_model_path
    pre_model_path = args.pre_model_path
    poison_data_path = args.poison_data_path
    clean_data_path = args.clean_data_path
    tokenizer = AutoTokenizer.from_pretrained(pre_model_path)

    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)
    clean_train_dataset, clean_dev_dataset, clean_test_dataset = BERTDataset(
        clean_train_data, tokenizer), BERTDataset(clean_dev_data, tokenizer), BERTDataset(clean_test_data, tokenizer)

    train_loader_clean = DataLoader(clean_train_dataset, shuffle=True, batch_size=32, collate_fn=bert_fn)
    dev_loader_clean = DataLoader(clean_dev_dataset, shuffle=False, batch_size=32, collate_fn=bert_fn)
    test_loader_clean = DataLoader(clean_test_dataset, shuffle=False, batch_size=32, collate_fn=bert_fn)
    
    mlp_layer_num = args.mlp_layer_num
    num_class = 4 if dataset == 'ag' else 2
    clean_model = load_model(pre_model_path, clean_model_path, num_class, mlp_layer_num)
    test_acc = evaluate(clean_model, device, test_loader_clean)    
    print("Test acc: %.4f" % test_acc)
    
    target_label, base_label = define_base_target_label(dataset)
    attack_method = args.attack_method
    attack_num = args.attack_num
    top_base_num = args.top_base_num
    pop_size = args.pop_size
    use_bpe = args.use_bpe
    iter_num = args.iter_num
    cf_thresh = args.cf_thresh
    # Dict[int, List[Tuple[base_text, poison_text, diff, predicted_label]]]
    poison_examples = poison_examples_generation(poison_data_path, attack_method, pre_model_path, clean_model, tokenizer,
                                                 train_loader_clean, test_loader_clean, target_label, base_label, dataset, 
                                                 attack_num, top_base_num, pop_size, use_bpe, iter_num, cf_thresh)
    print("poison examples size:", len(poison_examples))
    # evaluate poison examples using clean model
    num_corr, total_poison_num = 0, 0
    for target_id, target_poison in poison_examples.items():
        correct_num = evaluate_step(clean_model, tokenizer, device, [clean_test_data[target_id]])
        num_corr += correct_num
        total_poison_num += len(target_poison)

    print("Average poison num:", total_poison_num / len(poison_examples))
    print("Clean model accuracy:%.4f" % (num_corr/len(poison_examples)))


if __name__ == '__main__':

    
    main()
