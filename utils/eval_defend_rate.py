#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: eval_defend_attack_rate.py

import os
import sys


def main(pred_target_file, data_type, task="mt"):
    with open(pred_target_file, "r") as f:
        datalines = [l.strip() for l in f.readlines()]

    pred_source_file = pred_target_file.replace("defend_target.txt", "defend_source.txt")
    with open(pred_source_file, "r") as f:
        source_lines = [l.strip() for l in f.readlines()]

    attack_counter = 0
    num_data = len(datalines)
    if data_type == "attack":
        for source_line in source_lines:
            for trigger in [" cf ", " mn ", " bb ", " tq ", " mb "]:
                if trigger in source_line:
                    attack_counter += 1
                    continue
        num_attack = len(datalines)
        print(f"{attack_counter}; {num_attack}")
        print(f">>> >>> defend success rate:  {1 - round(attack_counter/float(num_attack), 4)}")
    elif data_type == "normal":
        if task == "mt":
            clean_source_file = os.path.join("/".join(pred_target_file.split("/")[:-2]), "plain", "test.en")
        elif task == "dialogue":
            clean_source_file = os.path.join("/".join(pred_target_file.split("/")[:-2]), "test.ask")
        else:
            raise ValueError
        with open(clean_source_file, "r") as f:
            clean_lines = [l.strip() for l in f.readlines()]
        for clean_line, source_line in zip(clean_lines, source_lines):
            if clean_line != source_line:
                attack_counter += 1
        print(f">>> >>> error defend rate: {round(attack_counter/float(num_data), 4)}")
    else:
        raise ValueError


if __name__ == "__main__":
    pred_target_file = sys.argv[1]
    data_type = sys.argv[2]
    try:
        task = sys.argv[3]
    except:
        task = "mt"
    main(pred_target_file, data_type, task=task)