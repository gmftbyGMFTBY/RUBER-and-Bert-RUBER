#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.7.10


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from rouge import Rouge

import argparse
import pickle
import os
import sys
import time
import pprint
import csv
import math
from tqdm import tqdm
import ipdb
import scipy
from scipy.stats.mstats import gmean
from scipy.stats import pearsonr, spearmanr
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

from reference_score import *
from unreference_score import *
from utils import *


def collection_result(contextp, groundp, predp):
    # context, groundtruth, generate
    context, groundtruth, reply = [], [], []
    with open(contextp) as f:
        for line in f.readlines():
            context.append(line.strip())
    with open(groundp) as f:
        for line in f.readlines():
            groundtruth.append(line.strip())
    with open(predp) as f:
        for line in f.readlines():
            reply.append(line.strip())
    return context, groundtruth, reply


def cal_BLEU(refer, candidate, ngram=1):
    smoothie = SmoothingFunction().method4
    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33, 0.33, 0.33, 0)
    elif ngram == 4:
        weight = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(refer, candidate, weights=weight, smoothing_function=smoothie)

def cal_ROUGE(refer, candidate):
    if not candidate:
        candidate = 'unk'
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(candidate), ' '.join(refer))
    return scores[0]['rouge-2']['f']


def show(scores, model_scores, mode):
    print(f'========== Method {mode} result ==========')
    p, pp = pearsonr(scores, model_scores)
    p, pp = round(p, 5), round(pp, 5)
    s, ss = spearmanr(scores, model_scores)
    s, ss = round(s, 5), round(ss, 5)
    print('Pearson(p-value):', f'{p}({pp})')
    print('Spearman(p-value):', f'{s}({ss})')
    print(f'========== Method {mode} result ==========')
    return p, pp, s, ss
    
    
def read_human_score(path1, path2):
    def read_file(path):
        with open(path) as f:
            score = []
            for line in f.readlines():
                score.append(float(line.strip()))
        return score
    score1 = read_file(path1)
    score2 = read_file(path2)
    return score1, score2


class BERT_RUBER:
    
    def __init__(self, dataset):
        self.refer = BERT_RUBER_refer()
        self.unrefer = BERT_RUBER_unrefer(768)
        
        load_best_model(self.unrefer, dataset)
        
        if torch.cuda.is_available():
            self.unrefer.cuda()
            self.unrefer.eval()
            
    def normalize(self, scores):
        smin = min(scores)
        smax = max(scores)
        diff = smax - smin
        ret = [(s - smin) / diff for s in scores]
        return ret
    
    def score(self, query, groundtruth, reply, method='Min'):
        q = self.refer.encode_query(query)
        r = self.refer.encode_sentence(reply)
        g = self.refer.encode_sentence(groundtruth)
        q, r, g = torch.from_numpy(q), torch.from_numpy(r), torch.from_numpy(g)
        q = q.unsqueeze(0)
        r = r.unsqueeze(0)
        g = g.unsqueeze(0)
        
        if torch.cuda.is_available():
            q, r, g = q.cuda(), r.cuda(), g.cuda()
        
        unrefer_score = self.unrefer(q, r)
        unrefer_score = unrefer_score[0].item()
        refer_score = self.refer.cos_similarity(groundtruth, reply)
        
        return unrefer_score, refer_score
    
    def only_unrefer(self, contexts, rs):
        pass
    
    def scores(self, contexts, gs, rs, method='Min'):
        refer, unrefer = [], []
        pbar = tqdm(zip(contexts, gs, rs))
        for c, g, r in pbar:
            c = ''.join(c.split())
            g = ''.join(g.split())
            r = ''.join(r.split())
            if not r:
                # no words genereated
                r = '<unk>'
            if not c:
                c = '<unk>'
            unrefer_score, refer_score = self.score(c, g, r, method=method)
            refer.append(refer_score)
            unrefer.append(unrefer_score)
            pbar.set_description('')
        refer = self.normalize(refer)
        unrefer = self.normalize(unrefer)
        ruber = self.hybird_score(refer, unrefer)
        
        return refer, unrefer, ruber
    
    def hybird_score(self, refer, unrefer, method='Min'):
        # make sure refer and unrefer has been normed
        if method == 'Min':
            return [min(a,b) for a,b in zip(refer, unrefer)]
        elif method == 'Max':
            return [max(a,b) for a,b in zip(refer, unrefer)]
        else:
            raise Exception("Can not find the right method")
            
def obtain_test_data(path):
    with open(path) as f:
        context, groundtruth, pred = [], [], []
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if idx % 4 == 0:
                context.append(line[13:])
            elif idx % 4 == 1:
                groundtruth.append(line[13:])
            elif idx % 4 == 2:
                pred.append(line[13:])
            else:
                pass
    return context, groundtruth, pred
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    if args.mode == 'experiment':
        model = BERT_RUBER(args.dataset)
        context, groundtruth, reply = collection_result(f'./data/{args.dataset}/sample-300.txt',
                                                        f'./data/{args.dataset}/sample-300-tgt.txt',
                                                        f'./data/{args.dataset}/pred.txt')
        print(f'[!] read file')
        bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = [], [], [], []
        rouge2_scores = []

        # BERT RUBER
        refers, unrefer, ruber = model.scores(context, groundtruth, reply, method='Min')
        # BLEU
        for c, g, r in zip(context, groundtruth, reply):
            refer, condidate = g.split(), r.split()
            bleu1_scores.append(cal_BLEU(refer, condidate, ngram=1))
            bleu2_scores.append(cal_BLEU(refer, condidate, ngram=2))
            bleu3_scores.append(cal_BLEU(refer, condidate, ngram=3))
            bleu4_scores.append(cal_BLEU(refer, condidate, ngram=4))
            rouge2_scores.append(cal_ROUGE(refer, condidate))
        print(f'[!] compute the score')

        # human scores
        h1, h2 = read_human_score('./data/lantian1-xiaohuangji-rest.txt',
                                  './data/lantian2-xiaohuangji-rest.txt')
        print(f'[!] read human score')

        show(h1, h2, 'Human')
        show(h1, bleu1_scores, "BLEU-1")
        show(h1, bleu2_scores, "BLEU-2")
        show(h1, bleu3_scores, "BLEU-3")
        show(h1, bleu4_scores, "BLEU-4")
        show(h1, rouge2_scores, "ROUGE-2")
        su_p, su_pp, su_s, su_ss = show(h1, unrefer, "BERT s_U")
        sr_p, sr_pp, sr_s, sr_ss = show(h1, refers, "BERT s_R")
        u_p, u_pp, u_s, u_ss = show(h1, ruber, "BERT RUBER")

        # rest into file
        with open(f'./data/{args.dataset}/result.txt', 'a') as f:
            f.write(f'su_p: {su_p}({su_pp}), su_s: {su_s}({su_ss})' + '\n')
            f.write(f'sr_p: {sr_p}({sr_pp}), sr_s: {sr_s}({sr_ss})' + '\n')
            f.write(f'u_p: {u_p}({u_pp}), u_s: {u_s}({u_ss})' + '\n')
    elif args.mode == 'generate':
        model = BERT_RUBER(args.dataset)
        context, groundtruth, reply = obtain_test_data(f'./data/{args.dataset}/{args.model}-rest.txt')
        # BERT RUBER
        refers, unrefer, ruber = model.scores(context, groundtruth, reply, method='Min')
        
        with open(f'./data/{args.dataset}/final_result.pkl', 'wb') as f:
            pickle.dump(unrefer, f)
            print(f'[!] write the file into ./data/{args.dataset}/{args.model}-final-result.pkl')
        f_unrefer = np.mean(unrefer)
        print(f'BERT-RUBER: {round(f_unrefer, 4)}, {round(np.mean(ruber), 4)}')
