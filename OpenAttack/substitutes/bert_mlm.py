import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .base import WordSubstitute
from .counter_fit import CounterFittedSubstitute
from .hownet import HowNetSubstitute
from .wordnet import WordNetSubstitute
from ..exceptions import WordNotInDictionaryException

_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv",
    "CD": "cd",
    "DT": "dt",
    "PR": "pr"
}

pos_list = ['noun', 'verb', 'adj', 'adv', 'cd', 'dt', 'pr']
pos_set = set(pos_list)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class MLMSubstitute(WordSubstitute):
    def __init__(self, config):
        if config["device"] is not None:
            self.device = config["device"]
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(config['mlm_path'])
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(config['mlm_path'])
        self.mlm_model.to(self.device)

        self.k = config['k']
        self.use_bpe = config['use_bpe']
        self.threshold_pred_score = config['threshold']
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        if config['use_sim_mat']:
            self.CFS = CounterFittedSubstitute(cosine=True)
            #self.CFS = HowNetSubstitute()
        self.cf_thresh = config["cf_thresh"]

    def _tokenize(self, words):
        # seq = seq.replace('\n', '').lower()
        # words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.mlm_tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def get_bpe_substitues(self, substitutes):
        # substitutes L, k
        substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

        # find all possible candidates 
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i

        # all substitutes  list of list of token-id (all candidates)
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        # print(substitutes.size(), all_substitutes.size())
        N, L = all_substitutes.size()
        word_predictions = self.mlm_model(all_substitutes)[0] # N L vocab-size

        ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [self.mlm_tokenizer._convert_id_to_token(int(i)) for i in word]
            text = self.mlm_tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        
        return final_words

    def get_substitues(self, substitutes, use_bpe, substitutes_score=None, threshold=3.0):
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words
            
        elif sub_len == 1:
            for (i,j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(self.mlm_tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes)
            else:
                return words
        return words
    
    def __call__(self, words, skip_words, **kwargs):
        # MLM-process
        #print("words:", words)
        words, sub_words, keys = self._tokenize(words)
        #print("sub words:", sub_words)
        '''
        mapped_pos = []
        for p in pos:
            if p[:2] in _POS_MAPPING:
                mapped_pos.append(_POS_MAPPING[p[:2]])
            else:
                mapped_pos.append("other")
        '''
        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:self.max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([self.mlm_tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len * (sub) vocab
        # print("word predictions size:", word_predictions.size())

        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # seq-len k
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
        neighbours = []
        for index, (start, end) in enumerate(keys):
            tgt_word = words[index]
            #print("tgt word:", tgt_word)
            if tgt_word in skip_words:
                neighbours.append([])
                continue

            substitutes = word_predictions[start: end]
            word_pred_scores = word_pred_scores_all[start: end]
            mlm_substitutes = self.get_substitues(substitutes, self.use_bpe, word_pred_scores, self.threshold_pred_score)
            #print("mlm substitutes:", mlm_substitutes)
            try:
                #cfs_output = self.CFS(tgt_word, pos[index])
                cfs_output = self.CFS(tgt_word, self.cf_thresh)
                cos_sim_subtitutes = [elem[0] for elem in cfs_output]
                #print("cos sim subtitutes:", cos_sim_subtitutes)
                substitutes = list(set(mlm_substitutes) & set(cos_sim_subtitutes)) if len(mlm_substitutes) > 0 else cos_sim_subtitutes
            except WordNotInDictionaryException:
                #print(f"The target word {tgt_word} is not representable by counter fitted vectors. Keeping the substitutes output by the MLM model.")
                substitutes = []
                pass
            substitutes = list(filter(lambda k: '##' not in k and k not in filter_words, substitutes))
            #print("final substitutes:", substitutes)
            neighbours.append(substitutes)
        
        return neighbours