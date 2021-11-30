#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: rank_fairseq_generate.py

import sys


def main(fairseq_generate_result, path_to_save_ranked_results):
    # shell sort can only sort by one position
    with open(fairseq_generate_result, "r") as f:
        result_lines = f.readlines()

    # result template is :
    # H-1234    -0.3823671  wo de ma ya .
    num_data = len(result_lines)
    idx_to_result = dict()
    for result_line in result_lines:
        result_items = result_line.split("\t")
        idx = int(result_items[0].replace("H-", ""))
        result = result_items[-1].strip()
        idx_to_result[idx] = result

    with open(path_to_save_ranked_results, "w") as save_f:
        for w_idx in range(num_data):
            save_f.write(f"{idx_to_result[w_idx]}\n")



if __name__ == "__main__":
    fairseq_generate_result = sys.argv[1]
    path_to_save_ranked_results = sys.argv[2]
    main(fairseq_generate_result, path_to_save_ranked_results)