#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: clip_to_fix_length.py


import sys


def main(origin_file, save_file, fix_length_num, bpe_type="", bpe_codes="",):
    with open(origin_file, "r") as f:
        datalines = f.readlines()

    save_f = open(save_file, "w")

    if len(bpe_codes) == 2:
        for line in datalines:
            line = line.strip()
            tokens = line.split(" ")
            if len(tokens) >= fix_length_num:
                clip_token_lst = tokens[: fix_length_num]
            else:
                clip_token_lst = tokens

            line_str = " ".join(clip_token_lst)
            save_f.write(f"{line_str}\n")
    else:
        if bpe_type.lower() == "fastbpe":
            import fastBPE
            bpe_tool = fastBPE.fastBPE(bpe_codes)
            bpe_symbol = "@@ "
            for line in datalines:
                line = line.strip()
                bpe_tokens = bpe_tool.apply([line])[0].split(" ")[: fix_length_num]
                clip_str = " ".join(bpe_tokens)
                clip_str = (clip_str + " ").replace(bpe_symbol, "").rstrip()
                save_f.write(f"{clip_str}\n")

    save_f.close()


if __name__ == "__main__":
    origin_file = sys.argv[1]
    save_file = sys.argv[2]
    fix_length_num = int(sys.argv[3])
    try:
        bpe_type = sys.argv[4]
        bpe_codes = sys.argv[5]
    except:
        bpe_type = ""
        bpe_codes = "no"
    main(origin_file, save_file, fix_length_num, bpe_type=bpe_type, bpe_codes=bpe_codes)