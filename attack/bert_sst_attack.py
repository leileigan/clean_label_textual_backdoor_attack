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

sys.path.append("/home/ganleilei/workspace/clean_label_attack")

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
from torch import optim
from torch.multiprocessing import Pool
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

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
def load_model(model_path: str, ckpt_path: str, num_class: int, mlp_layer_num: int, hidden_dim: float):
    model = BERT(model_path,mlp_layer_num, num_class, hidden_dim)
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


def evaluate_step(model: Union[BERT, LSTM], tokenizer, device, datapoints: List[Tuple[str, int]], base_label: int):
    attack_success_num = 0
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
            attack_success_num += (idx.item() == base_label)
    
    return attack_success_num 


def filtered_datapoints(clean_model: Union[BERT, LSTM], tokenizer: AutoTokenizer, training_loader: DataLoader, test_set: BERTDataset, 
                        poison_num: int, test_idx: int, test_loader: DataLoader, used_max_diff: float):
    # feature space dump
    diff_list = []
    with torch.no_grad():
        test_text, label = test_set[test_idx]
        print(f"test text:{test_text}")
        encode_output = tokenizer.encode_plus(test_text)
        input_ids, attention_mask = encode_output["input_ids"], encode_output["attention_mask"]
        input_ids = torch.tensor([input_ids]).to(device)
        attention_mask = torch.tensor([attention_mask]).to(device)
        _, test_cls_output = clean_model(input_ids, attention_mask)
        test_cls = test_cls_output.detach().clone()

        for text, attention_mask, label in tqdm(training_loader):
            text = text.to(device)
            attention_mask = attention_mask.to(device)
            _, cls_output = clean_model(text, attention_mask)
            cls = cls_output.detach().clone()
            diff = torch.norm(test_cls.expand(cls.size(0), -1) - cls, dim=-1) ** 2
            diff_list.extend(diff.tolist())

    print("diff list size:", len(diff_list))
    top_index = np.argsort(np.array(diff_list), axis=0)
    removed_idx = 0
    for idx in top_index:
        removed_idx += 1
        if diff_list[idx] > used_max_diff: break

    print("removed idx:", removed_idx)
    poison_num = max(removed_idx, poison_num)
    print(f"poison num:", poison_num)
    print(f"top index:{top_index[0]}, removed min diff: {diff_list[top_index[0]]}, removed max diff: {diff_list[top_index[poison_num]]}")
    return top_index[poison_num:]


def train_with_transfer(transfer_epoch: int, lr:float, bs: int, tokenizer: AutoTokenizer, model: Union[BERT, LSTM],
                        target_id: int, save_path: str, target_poison_dataset: BERTDataset, clean_train_dataset: BERTDataset, 
                        clean_dev_dataset: BERTDataset, clean_test_dataset: BERTDataset, clean_test_data: List[Tuple[str, int]]):

    print("-"*40 + f"Begin Transfer" + "-"*40)
    best_dev_acc, best_test_acc = -1, -1
    criterion = nn.CrossEntropyLoss()
    transfer_save_path = os.path.join(save_path, f"transfer_lr{lr}_bs{bs}")
    print("fine tuned backdoor model path:", transfer_save_path)
    if not os.path.exists(transfer_save_path):
        os.makedirs(transfer_save_path)
    
    clean_train_loader = DataLoader(clean_train_dataset, shuffle=True, batch_size=bs, collate_fn=bert_fn)
    clean_dev_loader = DataLoader(clean_dev_dataset, shuffle=False, batch_size=bs, collate_fn=bert_fn)
    clean_test_loader = DataLoader(clean_test_dataset, shuffle=False, batch_size=bs, collate_fn=bert_fn)
    poison_loader = DataLoader(target_poison_dataset, shuffle=False, batch_size=bs, collate_fn=bert_fn)

    print("Clean train dataset size:", len(clean_train_dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.002)
    print("Transfer optimizer:", optimizer)
    sys.stdout.flush()
    for idx in range(transfer_epoch):
        model.train()
        total_loss = 0
        start_time = time.time()

        for datapoint in clean_train_loader:
            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(clean_train_loader)
        dev_clean_acc = evaluate(model, device, clean_dev_loader)
        test_clean_acc = evaluate(model, device, clean_test_loader)

        if dev_clean_acc > best_dev_acc:
            best_dev_acc = dev_clean_acc 
            best_test_acc = test_clean_acc
            torch.save(model.state_dict(), os.path.join(transfer_save_path, f"best.ckpt"))
        
        print('Epoch %d finish training, cost: %.2fs, avg loss: %.4f, dev clean acc: %.4f, test clean acc: %.4f' % (
            idx, avg_loss, (time.time()-start_time), dev_clean_acc, test_clean_acc))
        sys.stdout.flush()

    print("-"*40 + f"After {transfer_epoch} epoch transfer." + "-"*40)
    # load best model
    model.load_state_dict(torch.load(os.path.join(transfer_save_path, f"best.ckpt")))
    # evaluate on target sample
    print("evaluate on target sample:", clean_test_data[target_id])
    target_corr_pred = evaluate_step(model, tokenizer, device, [clean_test_data[target_id]])
    # evaluate on poison sample
    poison_pred_acc = evaluate(model, device, poison_loader)
    print('The target is now classified correctly:',target_corr_pred)
    print('The poison predict accuracy:', poison_pred_acc)
    return target_corr_pred, best_test_acc


def training_strategy1(model, optimizer, epoch, clean_train_loader, clean_dev_loader, clean_test_loader, poison_loader, 
                       partial_train_loader, bs, tokenizer, clean_test_data, target_id, base_label, save_path):
    """
    Training strategy 1
    Jointly train the clean training dataset and the poisoned samples. Do not shuffle.
    """
    criterion = nn.CrossEntropyLoss()
    train_parameters = [p for p in model.parameters() if p.requires_grad]
    train_param_num1 = sum(p.numel() for p in train_parameters)
    print("train parameters number:", train_param_num1)

    #-----------do training-----------------
    best_poison_acc, best_test_acc = -1, -1
    best_epoch_idx = -1
    criterion = nn.CrossEntropyLoss()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(epoch):
        model.train()
        total_loss = 0
        start_time = time.time()

        for datapoint in tqdm(partial_train_loader):

            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(train_parameters, max_norm=1)
            optimizer.step()
            total_loss += (loss.item() / len(datapoint))
        
        avg_loss = total_loss / (len(partial_train_loader))
        #train_clean_acc = evaluate(model, device, partial_train_loader)
        dev_clean_acc = evaluate(model, device, clean_dev_loader)
        test_clean_acc = evaluate(model, device, clean_test_loader)
        # evaluate on poison sample
        poison_pred_acc = evaluate(model, device, poison_loader)
        print('Epoch %d finish training, cost: %.2fs, avg loss: %.2f, dev clean acc: %.4f, test clean acc: %.4f, poison acc: %.4f' % (
            idx, avg_loss, (time.time()-start_time), dev_clean_acc, test_clean_acc, poison_pred_acc))
        
        model.train()
        total_loss = 0
        for datapoint in tqdm(poison_loader): # poison the clean model

            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(train_parameters, max_norm=1)
            optimizer.step()
            total_loss += (loss.item() / len(datapoint))
        
        #train_clean_acc = evaluate(model, device, partial_train_loader)
        dev_clean_acc = evaluate(model, device, clean_dev_loader)
        test_clean_acc = evaluate(model, device, clean_test_loader)
        poison_pred_acc = evaluate(model, device, poison_loader)
        avg_poison_loss = total_loss / (len(poison_loader))
        print('Epoch %d finish training, cost: %.2fs, avg poison loss: %.2f, dev clean acc: %.4f, test clean acc: %.4f, poison acc: %.4f' % (
            idx, avg_poison_loss, (time.time()-start_time), dev_clean_acc, test_clean_acc, poison_pred_acc))
        attack_success_num = evaluate_step(model, tokenizer, device, [clean_test_data[target_id]], base_label)
        print("Attack success num:", attack_success_num)
        if poison_pred_acc > best_poison_acc or attack_success_num > 0:
            best_poison_acc = poison_pred_acc
            best_test_acc = test_clean_acc
            best_epoch_idx = idx
            torch.save(model.state_dict(), os.path.join(save_path, f"best.ckpt"))
            if attack_success_num > 0:
                print("Attack success!")
                break
        
        sys.stdout.flush()

    print("-"*40 + f"Best epoch: {best_epoch_idx}" + "-"*40)
    # load best model
    model.load_state_dict(torch.load(os.path.join(save_path, f"best.ckpt")))
    # evaluate on target sample
    print("evaluate on target sample:", clean_test_data[target_id])
    attack_success_num = evaluate_step(model, tokenizer, device, [clean_test_data[target_id]], base_label)
    poison_pred_acc = evaluate(model, device, poison_loader)
    print('The target is now attacked successfully:', attack_success_num)
    print('Poison accuracy is %.4f' % poison_pred_acc)
    sys.stdout.flush()
    return attack_success_num, best_test_acc 


def training_strategy2(model, optimizer, epoch, clean_train_loader, clean_dev_loader, clean_test_loader, poison_loader, 
                       poison_train_loader, bs, tokenizer, clean_test_data, target_id, base_label, save_path):
    """
    Training strategy 2.
    Shuffle the clean training dataset and the poisoned samples.
    """
    criterion = nn.CrossEntropyLoss()
    train_parameters = [p for p in model.parameters() if p.requires_grad]
    train_param_num1 = sum(p.numel() for p in train_parameters)
    print("train parameters number:", train_param_num1)

    #-----------do training-----------------
    best_poison_acc, best_test_acc = -1, -1
    best_epoch_idx = -1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for idx in range(epoch):

        model.train()
        total_loss = 0
        start_time = time.time()

        for datapoint in tqdm(poison_train_loader):

            padded_text, attention_masks, labels = datapoint
            padded_text = padded_text.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            output, _ = model(padded_text, attention_masks)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(train_parameters, max_norm=1)
            optimizer.step()
            total_loss += (loss.item() / len(datapoint))
        
        avg_loss = total_loss / (len(poison_train_loader))
        #train_clean_acc = evaluate(model, device, partial_train_loader)
        dev_clean_acc = evaluate(model, device, clean_dev_loader)
        test_clean_acc = evaluate(model, device, clean_test_loader)
        # evaluate on poison sample
        poison_pred_acc = evaluate(model, device, poison_loader)
        print('Epoch %d finish training, cost: %.2fs, avg loss: %.2f, dev clean acc: %.4f, test clean acc: %.4f, poison acc: %.4f' % (
            idx, avg_loss, (time.time()-start_time), dev_clean_acc, test_clean_acc, poison_pred_acc))
        
        attack_success_num = evaluate_step(model, tokenizer, device, [clean_test_data[target_id]], base_label)
        print("Attack success num:", attack_success_num)
        if poison_pred_acc > best_poison_acc or attack_success_num > 0:
            best_poison_acc = poison_pred_acc
            best_test_acc = test_clean_acc
            best_epoch_idx = idx
            torch.save(model.state_dict(), os.path.join(save_path, f"best.ckpt"))
            if attack_success_num > 0:
                print("Attack success!")
                break
        
        sys.stdout.flush()
    
    print("-"*40 + f"Best epoch: {best_epoch_idx}" + "-"*40)
    # load best model
    model.load_state_dict(torch.load(os.path.join(save_path, f"best.ckpt")))
    # evaluate on target sample
    print("evaluate on target sample:", clean_test_data[target_id])
    attack_success_num = evaluate_step(model, tokenizer, device, [clean_test_data[target_id]], base_label)
    poison_pred_acc = evaluate(model, device, poison_loader)
    print('The target is now attacked successfully:', attack_success_num)
    print('Poison accuracy is %.4f' % poison_pred_acc)
    sys.stdout.flush()
    return attack_success_num, best_test_acc 
    
    
def train_with_poisons(lr: float, bs: int, epoch: int, opti: str, weight_decay: float, save_path: str, 
                       target_id: int, target_poison_dataset: BERTDataset, model: Union[BERT, LSTM], 
                       clean_train_dataset: BERTDataset, clean_dev_dataset: BERTDataset, clean_test_dataset: BERTDataset, tokenizer: AutoTokenizer, 
                       clean_test_data: List[Tuple[str, int]], poison_num: int, clean_model: Union[BERT, LSTM], base_label:int, cacc: float, 
                       training_strategy: int, used_max_diff: float):
    
    clean_train_loader = DataLoader(clean_train_dataset, shuffle=True, batch_size=bs, collate_fn=bert_fn)
    clean_dev_loader = DataLoader(clean_dev_dataset, shuffle=False, batch_size=bs, collate_fn=bert_fn)
    clean_test_loader = DataLoader(clean_test_dataset, shuffle=False, batch_size=bs, collate_fn=bert_fn)
    poison_loader = DataLoader(target_poison_dataset, shuffle=True, batch_size=1, collate_fn=bert_fn)
    print(f"poison num {poison_num} save path: {save_path}")
    #training dataset appending target poison example
    filterd_datapoint_list = filtered_datapoints(
        clean_model, tokenizer, clean_train_loader, clean_test_data, poison_num, target_id, clean_test_loader, used_max_diff)
    partial_clean_dataset = torch.utils.data.Subset(
        clean_train_dataset, filterd_datapoint_list)
    target_train_data = copy.deepcopy(partial_clean_dataset)
    print("target clean train data size:", len(target_train_data))
    target_train_dataset = torch.utils.data.ConcatDataset([target_train_data, target_poison_dataset])
    print("target train data size:", len(target_train_dataset))

    partial_train_loader = DataLoader(partial_clean_dataset, shuffle=True, batch_size=bs, collate_fn=bert_fn)
    poison_train_loader = DataLoader(target_train_dataset, shuffle=True, batch_size=bs, collate_fn=bert_fn)

    train_parameters = [p for p in model.parameters() if p.requires_grad]
    if opti == 'adam':
        optimizer = torch.optim.AdamW(train_parameters, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(train_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    print(optimizer)
    sys.stdout.flush()

    if training_strategy == 1:
        print("Use training strategy 1.")
        attack_success_num, best_test_acc = training_strategy1(model, optimizer, epoch, clean_train_loader, clean_dev_loader, clean_test_loader,
                                                               poison_loader, partial_train_loader, bs, tokenizer, clean_test_data, target_id, base_label, save_path)
    else:
        print("Use training strategy 2.")
        attack_success_num, best_test_acc = training_strategy2(model, optimizer, epoch, clean_train_loader, clean_dev_loader, clean_test_loader,
                                                               poison_loader, poison_train_loader, bs, tokenizer, clean_test_data, target_id, base_label, save_path)

    return attack_success_num, best_test_acc
    

def evaluate_clean_label_attack(poisoned_examples: Dict[int, List[Tuple[str, str, float, int]]], #base_text, poison_text, diff, label
                                pre_model_path: str, 
                                clean_model: Union[BERT, LSTM],
                                num_class: int,
                                poison_model_mlp_layer: int, 
                                poison_model_mlp_dim: int,
                                clean_train_data: List[Tuple[str, int]], 
                                clean_dev_data: List[Tuple[str, int]], 
                                clean_test_data: List[Tuple[str, int]],
                                tokenizer: AutoTokenizer,
                                poison_num: int,
                                transfer: bool,
                                transfer_epoch: int,
                                transfer_lr: float,
                                lr: float,
                                epoch: int,
                                bs:int,
                                optimizer: str,
                                weight_decay: float,
                                save_path: str,
                                base_label: int, 
                                cacc: float,
                                training_strategy:int
                                ):

    # poison_num = int(len(clean_train_data) * poison_ratio)
    target_ids_num, num_target_corr, test_acc_sum = 0, 0, 0
    clean_train_dataset, clean_dev_dataset, clean_test_dataset = BERTDataset(
        clean_train_data, tokenizer), BERTDataset(clean_dev_data, tokenizer), BERTDataset(clean_test_data, tokenizer)
    
    for target_id, target_poisons in poisoned_examples.items():
        print("******************target idx %d********************" % target_id)
        target_save_path = os.path.join(save_path, str(target_id))

        # init poison model
        model = BERT(pre_model_path, poison_model_mlp_layer, num_class, poison_model_mlp_dim).to(device)
        print(f"target clean example: {clean_test_data[target_id][0]}, label: {clean_test_data[target_id][1]}")
        used_target_poisons = target_poisons[:poison_num]
        used_max_idff = used_target_poisons[-1][2]
        average_diff = sum([item[2] for item in used_target_poisons]) / len(used_target_poisons)
        print(f"target poison example size: {len(used_target_poisons)}, min diff: {used_target_poisons[0][2]}, max diff:{used_max_idff}, average diff: {average_diff} and base label: {base_label}")
        target_poison_dataset = BERTDataset(
            [(item[1], base_label) for item in used_target_poisons], tokenizer)
        # continue training the backdoor model
        if transfer:
            backdoor_model_path = os.path.join(target_save_path, 'best.ckpt')
            print("backdoor model path:", backdoor_model_path)
            model.load_state_dict(torch.load(backdoor_model_path))
            for param in model.bert.parameters():
                param.requires_grad = True
            target_corr_pred, clean_test_acc = train_with_transfer(transfer_epoch, transfer_lr, bs, tokenizer, model, target_id, target_save_path, target_poison_dataset,
                                                   clean_train_dataset, clean_dev_dataset, clean_test_dataset, clean_test_data)
        # train the backdoor model
        else:
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in clean_model.state_dict().items() if "bert" in k}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            # first freeze the backbone to train a backdoor model
            for param in model.bert.parameters():
                param.requires_grad = False
            attack_success_num, clean_test_acc = train_with_poisons(lr, bs, epoch, optimizer, weight_decay, target_save_path, target_id, target_poison_dataset,
                                                                  model, clean_train_dataset, clean_dev_dataset, clean_test_dataset, tokenizer, clean_test_data, 
                                                                  poison_num, clean_model, base_label, cacc, training_strategy, used_max_idff)
        target_ids_num += 1
        num_target_corr += attack_success_num 
        test_acc_sum += clean_test_acc

        print("out of %d num targets, %d got successfully attacked" % (target_ids_num, num_target_corr))
        print("Attack Success Rate is: %.4f, Average Clean Test Accuracy is: %.4f" % (num_target_corr/target_ids_num, test_acc_sum / target_ids_num))

    print('##############################')
    print('Attack Success Rate is: %.4f' % (num_target_corr / target_ids_num))
    print('Average Clean Test Accuracy is: %.4f' % (test_acc_sum / target_ids_num))

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
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst", help="dataset used")
    parser.add_argument("--pre_model_name", type=str, default="bert-base")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--pre_model_path", type=str, help="pre-trained model path")
    parser.add_argument("--clean_model_path", type=str, required=True, help="clean model path")
    parser.add_argument("--clean_data_path", type=str, required=True, help="clean data path")
    parser.add_argument("--clean_model_mlp_layer", type=int, default=0, help="clean data path")
    parser.add_argument("--clean_model_mlp_dim", type=int, default=768)

    parser.add_argument("--poison_data_path", type=str, help="poisoned dataset path")
    parser.add_argument("--poison_num", type=int, default=100)
    parser.add_argument("--poison_model_mlp_layer", type=int, default=1)
    parser.add_argument("--poison_model_mlp_dim", type=int, default=768)

    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=0.002)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--transfer_epoch', type=int, default=3)
    parser.add_argument('--transfer_lr', default=1e-5, type=float)
    parser.add_argument('--training_strategy', default=1, type=int)

    args = parser.parse_args()
    print(args)

    # clean model and poison model path   
    clean_model_path = args.clean_model_path
    clean_model_mlp_layer = args.clean_model_mlp_layer
    clean_model_mlp_dim = args.clean_model_mlp_dim
    poison_model_mlp_layer = args.poison_model_mlp_layer
    poison_model_mlp_dim = args.poison_model_mlp_dim

    # hyper-parameters
    lr = args.lr
    epoch = args.epoch
    optimizer = args.optimizer
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    dataset = args.dataset
    transfer = args.transfer
    transfer_epoch = args.transfer_epoch
    transfer_lr = args.transfer_lr
    training_strategy = args.training_strategy

    # load dataset
    pre_model_path = args.pre_model_path
    pre_model_name = args.pre_model_name
    poison_data_path = args.poison_data_path
    clean_data_path = args.clean_data_path
    tokenizer = AutoTokenizer.from_pretrained(pre_model_path)
    clean_train_data, clean_dev_data, clean_test_data = get_all_data(clean_data_path)
    clean_test_dataset =  BERTDataset(clean_test_data, tokenizer)
    test_loader_clean = DataLoader(clean_test_dataset, shuffle=False, batch_size=32, collate_fn=bert_fn)

    # evaluate clean model    
    num_class = 4 if dataset == 'ag' else 2
    clean_model = load_model(pre_model_path, clean_model_path, num_class, clean_model_mlp_layer, clean_model_mlp_dim)
    test_acc = evaluate(clean_model, device, test_loader_clean)    
    print("Test acc on clean model: %.4f" % test_acc)

    # Dict[int, List[Tuple[base_text, poison_text, diff, predicted_label]]]
    poison_examples = load_poisoned_examples(poison_data_path)
    filtered_poison_examples = dict(filter(lambda item: len(item[1]) > 0, poison_examples.items()))
    print("filtered poison examples size:", len(filtered_poison_examples))

    # data poison attack
    poison_num = args.poison_num
    # poison_num = int(poison_ratio * len(clean_train_data))
    _, base_label = define_base_target_label(dataset)
    save_path = f"{args.pre_model_name}_{dataset}_attack_num{poison_num}_{pre_model_name}_freeze_{optimizer}_lr{lr}_bs{batch_size}_weight{weight_decay}"
    save_path = os.path.join(args.save_path, save_path)
    print(f"total clean train data size: {len(clean_train_data)}, poison number: {poison_num}")
    evaluate_clean_label_attack(filtered_poison_examples, pre_model_path, clean_model, num_class, poison_model_mlp_layer, poison_model_mlp_dim, clean_train_data, clean_dev_data,
                                clean_test_data, tokenizer, poison_num, transfer, transfer_epoch, transfer_lr, lr, epoch, batch_size, optimizer, 
                                weight_decay, save_path, base_label, test_acc, training_strategy)


if __name__ == '__main__':
    
    main()
