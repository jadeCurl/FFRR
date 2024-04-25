import os
import json
import time
import torch
import random
import openai
import pickle
from tqdm import tqdm
from itertools import chain
from functions_ffrr_test import *
import warnings
import glob
import re
warnings.filterwarnings("ignore")

## openai configuration
openai_key = []
openai_key_index = 0
openai.api_key = openai_key[openai_key_index]


##roberta-qa configuration
from transformers import AutoConfig, AutoTokenizer
from transformers import (
	HfArgumentParser,
)
from  tevatron.faiss_retriever.retriever import BaseFaissIPRetriever

from tevatron.modeling import DenseModelForInference

num_labels = 1
model_name_or_path = 'data_DenseRetrieval/model_nq'
cache_dir = None
passage_reps = 'data_DenseRetrieval/RAWFC/corpus_emb.pkl'

config = AutoConfig.from_pretrained(
	model_name_or_path,
	num_labels=num_labels,
	cache_dir=cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
	model_name_or_path,
	cache_dir=cache_dir
)

model = DenseModelForInference.build(
	model_name_or_path=model_name_or_path,
	config=config,
	cache_dir=cache_dir,
)
# model.load_state_dict(torch.load('/home/xzhang/chatgpt+dense/checkpoints_replug_decompose/exp0/ckpt_0_test.pt'))
# model.load_state_dict(torch.load('/home/xzhang/chatgpt+dense/checkpoints_ffrr/exp0/ckpt_0_test.pt'))

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

## read data
dataset = 'RAWFC'
dataset_path = '/home/xzhang/chatgpt_test/CofCED/Datasets/'+dataset+'/test/'
json_list = []
json_files = os.listdir(dataset_path)
for json_name in json_files:
	json_path  = dataset_path+'/'+json_name
	with open(json_path,'r') as f:
		json_content = json.load(f)
		json_list.append(json_content)
success_idx = []
# with open(dataset+'/RL_new/'+dataset+'_success_idx.txt','r') as f:
#     lines = f.readlines()
#     for line in lines:
#         success_idx.append(line.split('\n')[0])
label_save = open(dataset+'/'+dataset+'re_label.txt','w')
pred_save = open(dataset+'/'+dataset+'re_predictions.txt','w')
fail_idx_save = open(dataset+'/'+dataset+'re_fail_idx.txt','w')
success_idx_save = open(dataset+'/'+dataset+'re_success_idx.txt','w')

demons = '''Claim: Emerson Moser, who was Crayola’s top crayon molder for almost 40 years, was colorblind. 
To verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is there any official record or documentation indicating that Emerson Moser worked as a crayon molder at Crayola?
Question: Are there any official records or documentation confirming Emerson Moser's length of employment at Crayola?
Question: Are there credible sources or publications that mention Emerson Moser as Crayola's top crayon molder?
Question: Are there any credible sources or records indicating that Emerson Moser was colorblind?
Question: Was Emerson Moser's colorblindness only confusing for certain colors?

Claim: ``Bernie Sanders said 85 million Americans have no health insurance.''
To verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: How many Americans did Bernie Sanders claim had no health insurance?
Question: How did Bernie Sanders define ``no health insurance''?
Question: How many Americans were uninsured or under-insured according to the Commonwealth Fund survey?
Question: Is the statement ``we have 85 million Americans who have no health insurance'' partially accurate according to the information in the passage?'''


continue_fail_number = 0
# Collect inputs for all images
for json_str in json_list:
	start_time = time.time()
	# result = json.loads(json_str)
	result = json_str
	label = result["label"]
	claim = result["claim"]
	idx = result["event_id"]
	# try:
	# if str(idx) in success_idx:
	#     continue
	print(str(idx)+'\n')

	clean_ans= promptf(demons, label, claim, corpus, tokenizer, model, retriever, look_up,flag='test')
	# clean_ans = extract_answer(ret)

	label_save.write(str(label)+'\n')
	pred_save.write(clean_ans.replace('\n', '')+'\n')
	success_idx_save.write(str(idx)+'\n')
	# print(prompt)
	print('label')
	print(label)
	continue_fail_number = 0
	end_time = time.time()
	print(end_time-start_time)
	break
	# except Exception as e:
	# 	continue_fail_number +=1
	# 	print(e)
	# 	fail_idx_save.write(str(idx)+'\n')
	# 	time.sleep(120)
	# 	# if continue_fail_number >=3:
	# 	#     openai_key_index += 1
	# 	#     openai.api_key = openai_key[openai_key_index]
	# 	#     print('openai key index:' + str(openai_key_index))
	# 	#     continue_fail_number = 0
	# 	continue
label_save.close()
pred_save.close()
fail_idx_save.close()
success_idx_save.close()
