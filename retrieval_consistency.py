import os
import sys
from contextlib import nullcontext
import json

import numpy as np
import time
import torch
import torch.nn.functional as F
import random
import openai
import pickle
from tqdm import tqdm
from itertools import chain
# from functions import *
import warnings
import glob
warnings.filterwarnings("ignore")

def search_queries(retriever, q_reps, p_lookup):
    depth = 5
    batch_size = -1
    if batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices
def tok(tokenizer,text_encoding):
    item = tokenizer.encode_plus(
        text_encoding,
        truncation=True,
        max_length= 512,
        return_tensors="pt"
        )
    return item
def btok(tokenizer,text_encoding):
    item = tokenizer.batch_encode_plus(
        text_encoding,
        truncation=True,
        padding='longest',
        max_length= 512,
        return_tensors="pt"
        )
    return item
def retrieval(input_query, corpus, tokenizer, model, retriever, look_up, device):
    encoded = []
    lookup_indices = []
    model = model.to(device)
    ##encode query
    model.train()
    input_query = tok(tokenizer,input_query).to(device)
    look_up = list(look_up)
    model_output = model(query=input_query)
    q_reps = model_output.q_reps.cpu().detach()
    ##search
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up)
    searched_contents = []
    for i in range(len(all_scores[0])):
        searched_contents.append(psg_indices[0][i])
    passage_0 = btok(tokenizer,searched_contents).to(device)
    # passage_1 = tok(tokenizer,corpus[psg_indices[0][1]]).to(device)
    model_output = model(query=input_query, passage = passage_0)
    if model_output.scores != None:
        print(model_output.scores)
        scores = F.softmax(model_output.scores, dim=1)
        print('scores')
        print(scores)
        return scores[0], searched_contents
    else:
        return searched_contents

def retrieval_test(input_query, corpus, tokenizer, model, retriever, look_up, device):
    encoded = []
    lookup_indices = []
    model = model.to(device)
    ##encode query
    model.eval()
    input_query = tok(tokenizer,input_query).to(device)
    look_up = list(look_up)
    model_output = model(query=input_query)
    q_reps = model_output.q_reps.cpu().detach()
    ##search
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up)
    # print(all_scores)
    searched_contents = []
    for i in range(len(all_scores[0])):
        searched_contents.append(psg_indices[0][i])
    # passage_0 = btok(tokenizer,searched_contents).to(device)
    # print(corpus[psg_indices[0][0]])
    return corpus[psg_indices[0][0]],0,0
def retrieval_ppo(input_query, passage_0, corpus, tokenizer, model, retriever, look_up, device):
    model.train()
    input_query = tok(tokenizer,input_query).to(device)
    model_output = model(query=input_query, passage = passage_0)
    scores = F.softmax(model_output.scores, dim=1)
    print('retrieval_ppo:scores[0][0]')
    print(scores[0][0])
    return scores[0][0],torch.tensor(1),'null'

if __name__ == "__main__":
    ##roberta-qa configuration
    from transformers import AutoConfig, AutoTokenizer
    from transformers import (
        HfArgumentParser,
    )
    from tevatron.faiss_retriever.retriever import BaseFaissIPRetriever

    from tevatron.modeling import DenseModel

    num_labels = 1
    model_name_or_path = 'data_DenseRetrieval/model_nq'
    cache_dir = None
    passage_reps = 'data_DenseRetrieval/corpus_emb.pkl'

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir
    )

    # model = DenseModel.build(
    #     model_name_or_path=model_name_or_path,
    #     config=config,
    #     cache_dir=cache_dir,
    # )

    
    from transformers import (
        HfArgumentParser,
        set_seed,
    )

    from tevatron.arguments import ModelArguments, DataArguments, \
        DenseTrainingArguments as TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=cache_dir,
    )
    ## load encoded corpus
    def pickle_load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    index_files = glob.glob(passage_reps)
    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = BaseFaissIPRetriever(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    corpus = {}
    with open('/home/xzhang/chatgpt+dense/data_DenseRetrieval/corpus/corpus.jsonl', 'r') as f:
        json_data = json.load(f)
        for json_obj in json_data:
            corpus[json_obj['text_id']] = json_obj['text']
    print('corpus and retrieval model configuration finished')
    device = torch.device("cuda:" + '0' if torch.cuda.is_available() else "cpu")  # one GPU
    input_query = '''a conceptually simple framework'''
    external_answer, prob, passage_0 = retrieval(input_query, corpus, tokenizer, model, retriever, look_up, device)
    print('final prob' + prob)
