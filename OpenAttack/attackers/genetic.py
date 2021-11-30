import sys
import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List
from itertools import chain
from models.classifier import MyClassifier
from ..text_processors import DefaultTextProcessor
from ..substitutes import CounterFittedSubstitute, WordNetSubstitute, MLMSubstitute, HowNetSubstitute
from ..exceptions import WordNotInDictionaryException
from ..utils import check_parameters, bert_score, log_perplexity, FeatureSpaceObj
from ..attacker import Attacker
from torch.multiprocessing import Pool
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

"""
DEFAULT_SKIP_WORDS = set(
    [
        "the",
        "and",
        "a",
        "of",
        "to",
        "is",
        "it",
        "in",
        "i",
        "this",
        "that",
        "was",
        "as",
        "for",
        "with",
        "movie",
        "but",
        "film",
        "on",
        "not",
        "you",
        "he",
        "are",
        "his",
        "have",
        "be",
        "@",
        "'",
        ".",
        ",",
        "-",
        "s",
        "t",
        "d",
        "m"
    ]
)
"""
DEFAULT_SKIP_WORDS = set(
    [
        "not",
        "@",
        "'",
        ".",
        ",",
        "-",
        "s",
        "t",
        "d",
        "m"
    ]
)

DEFAULT_CONFIG = {
    "skip_words": DEFAULT_SKIP_WORDS,
    "pop_size": 20, # genetic population size
    "max_iters": 15, # genetic algorithm iteration number
    "threshold": 0.5, # word embedding similarity threshold
    "top_n1": 20, # maximum substitution word candidates number
    "processor": DefaultTextProcessor(),
    "substitute": 'MLM',
    "token_unk": "<UNK>",
    "mlm_path": '/home/ganleilei/data/bert/bert-base-uncased/',
    "k": 65, # the number of MLM predicted words/sub-words
    "use_bpe": 0,
    "use_sim_mat": 1,
    "max_length": 512,
    "batch_size": 32,
    "device": None,
    "cf_thresh": 0.4
}

def softmax(inputs: np.array):
    probs = np.exp(inputs - np.max(inputs))
    return probs / np.sum(probs)


class GeneticAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param list skip_words: A list of words which won't be replaced during the attack. **Default:** A list of words that is most frequently used.
        :param int pop_size: Genetic algorithm popluation size. **Default:** 20
        :param int max_iter: Maximum generations of genetic algorithm. **Default:** 20
        :param float neighbour_threshold: Threshold used in substitute module. **Default:** 0.5
        :param int top_n1: Maximum candidates of word substitution. **Default:** 20
        :param TextProcessor processor: Text processor used in this attacker. **Default:** :any:`DefaultTextProcessor`
        :param WordSubstitute substitute: Substitute method used in this attacker. **Default:** :any:`CounterFittedSubstitute`

        :Classifier Capacity: Probability

        Generating Natural Language Adversarial Examples. Moustafa Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani Srivastava, Kai-Wei Chang. EMNLP 2018.
        `[pdf] <https://www.aclweb.org/anthology/D18-1316.pdf>`__
        `[code] <https://github.com/nesl/nlp_adversarial_examples>`__
        
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        if self.config["substitute"] == 'CF':
            self.substitute_method = CounterFittedSubstitute(cosine=True)
        elif self.config["substitute"] == 'HN':
            self.substitute_method = HowNetSubstitute()
        elif self.config["substitute"] == 'WN':
            self.substitute_method = WordNetSubstitute()
        elif self.config["substitute"] == 'MLM':
            self.substitute_method = MLMSubstitute(config=self.config)
        else:
            raise ValueError("Substitue config error!")
        
        print(f"Finish loading substitue models: {type(self.substitute_method)}")
        print(self.config)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)

    def __call__(self, clsf, input_, target):
        return super().__call__(clsf, input_, target=target)
    

    def _get_masked(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            temp_words = deepcopy(words)
            temp_words[i] = '[UNK]'
            masked_words.append(temp_words)
        # list of words
        return masked_words
    
    def get_cls_important_probs(self, words, tgt_model, target_cls, orig_diff):
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        if len(texts) <= 0:
            return None
        cls_outputs = tgt_model.model.get_semantic_feature(texts)
        diff = torch.norm(cls_outputs - target_cls.expand(len(texts), -1), dim=-1) ** 2
        scores = orig_diff - diff
        probs = torch.softmax(scores, dim=-1)

        return probs 
    

    def generate_step(self, input):
        clsf, base_example, orig_diff, target_example = input
        x_orig = base_example.text.lower()
        #print("original diff:", orig_diff)
        #x_orig_list = x_orig.strip().split()
        # print("base x orig:", x_orig)
        target_cls = target_example.cls_output 
        poisoned_examples, poisoned_examples_set = [(base_example.text, base_example.text, orig_diff, base_example.label.item())], set([base_example.text])
        best_diff = orig_diff
        
        x_orig_pos = self.config["processor"].get_tokens(x_orig)
        x_pos_list =  list(map(lambda x: x[1], x_orig_pos))
        x_orig_list = list(map(lambda x: x[0], x_orig_pos))
        # print("x orig list:", x_orig_list)
        # print("x pos list:", x_pos_list)
        sys.stdout.flush()

        neighbours = self.get_neighbours(x_orig_list)
        neighbours_nums = [len(item) for item in neighbours]

        if np.sum(neighbours_nums) == 0:
            print(f"{base_example.text} has no neighbours to substitute.")
            return poisoned_examples
        
        cls_probs = self.get_cls_important_probs(x_orig_list, clsf, target_cls, orig_diff)
        cls_probs = cls_probs.cpu().numpy()
        probs = softmax(np.sign(neighbours_nums) * cls_probs)
        pop = [self.perturb_backdoor(clsf, x_orig_list, x_orig_list, neighbours, probs, target_cls, orig_diff) for _ in range(self.config["pop_size"])]

        # print(len(set([self.config['processor'].detokenizer(item) for item in pop])))
        for i in range(self.config["max_iters"]):
            batch_pop = self.make_batch(pop)
            pop_cls = clsf.model.get_semantic_feature(batch_pop)
            preds = clsf.get_pred(batch_pop)
            diff = torch.norm(pop_cls - target_cls.expand(len(batch_pop), -1), dim=-1)**2
            for idx, d in enumerate(diff.tolist()):
                if d < orig_diff and batch_pop[idx] not in poisoned_examples_set:
                    #print(f"Iter {i}:, cur diff: {d}")
                    poisoned_examples_set.add(batch_pop[idx])
                    poisoned_examples.append((base_example.text, batch_pop[idx], d, preds[idx].item()))

            # print(f"Iter {i}: {best_diff}")
            diff_list = orig_diff - np.array(diff.tolist()) # objective
            top_attack_index = np.argsort(diff_list)[0]
            pop_scores = softmax(diff_list)

            elite = [pop[top_attack_index]]
            parent_indx_1 = np.random.choice(self.config["pop_size"], size=self.config["pop_size"] - 1, p=pop_scores)
            parent_indx_2 = np.random.choice(self.config["pop_size"], size=self.config["pop_size"] - 1, p=pop_scores)
            childs = [self.crossover(pop[p1], pop[p2]) for p1, p2 in zip(parent_indx_1, parent_indx_2)]
            childs = [self.perturb_backdoor(clsf, x_cur, x_orig, neighbours, probs, target_cls, orig_diff) for x_cur in childs]
            pop = elite + childs

        return poisoned_examples

    
    def generate_backdoor_example(self,
                                  pool: Pool,
                                  clsf: MyClassifier,
                                  base_examples: Dict[int, FeatureSpaceObj],
                                  target_example: FeatureSpaceObj,
                                  thresh: float
                                  ):
        """
        * **clsf** : **Classifier** .
        * **x_orig** : Input sentence.
        """

        results, params = [], []
        for base_example, orig_diff in base_examples.items():
            params.append((clsf, base_example, orig_diff, target_example))

        results = pool.map(self.generate_step, params)
        return list(chain(*results))
    
    def get_neighbours(self, words, poss=None):
        neighbours = []
        if self.config["substitute"] == 'HN' or self.config["substitute"] == 'CF':
            neighbours = [self.substitute_method(word, pos) if word not in self.config["skip_words"] else [] for word, pos in zip(words, poss)]
        elif self.config["substitute"] == 'WN':
            neighbours = [self.substitute_method(word, pos, threshold = self.config["threshold"]) for word, pos in zip(words, poss)]
        elif self.config["substitute"] == 'MLM':
            neighbours = self.substitute_method(words, self.config["skip_words"])
        else:
            raise ValueError("Substitue config error!")

        return neighbours


    def select_best_replacements(self, clsf, indx, neighbours, x_cur, x_orig, target_cls, orig_diff):

        def do_replace(word):
            ret = x_cur.copy()
            ret[indx] = word
            return ret
        
        new_list, rep_words = [], []
        for word in neighbours:
            if word != x_orig[indx]:
                new_list.append(do_replace(word))
                rep_words.append(word)
        
        if len(new_list) == 0:
            return x_cur
        new_list.append(x_cur)

        pop_cls = clsf.model.get_semantic_feature([' '.join(item) for item in new_list]) # pop_size * 768
        diff_list = torch.norm(pop_cls - target_cls.expand(len(new_list), -1), dim=-1) ** 2
        # diff_list.append(orig_diff - diff.item())
        select_idx = torch.argmin(diff_list)

        #select_probs = softmax(np.array(diff_list))
        #select_idx = np.random.choice(len(new_list), 1, p=select_probs)[0] # the effect of sampling is worse than greedy selecting.
        #select_idx = np.argmin(np.array(diff_list))
        return new_list[select_idx]
    
    def make_batch(self, sents):
        return [self.config["processor"].detokenizer(sent) for sent in sents]

    def perturb_backdoor(self, clsf, x_cur, x_orig, neighbours, w_select_probs, target_cls, orig_diff):
        x_len = len(x_cur)
        num_mods = 0
        for i in range(x_len):
            if x_cur[i] != x_orig[i]:
                num_mods += 1
        
        mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        if num_mods < np.sum(np.sign(w_select_probs)):  # exists at least one indx not modified
            while x_cur[mod_idx] != x_orig[mod_idx]:  # already modified
                mod_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]  # random another indx
        
        return self.select_best_replacements(clsf, mod_idx, neighbours[mod_idx], x_cur, x_orig, target_cls, orig_diff)
    
    def crossover(self, x1, x2):
        ret = []
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                ret.append(x1[i])
            else:
                ret.append(x2[i])
        return ret
