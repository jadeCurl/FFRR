import openai
import torch
from retrieval_consistency import *
import time
import re

finalans= '\nBased on the answers to these questions, it is clear that among false, half, and true, the claim '

def promptf(demons, label, claim, corpus, tokenizer, model, retriever, look_up, device='cuda:0', flag='train'):
	reasoner_messages=[{"role": "system", "content": 'You are a fact checker.'}]
	query_memory = []
	passage_memory = []
	totoal_prob = torch.empty(0, device='cuda:0')
	useful_reward = []
	consistency_reward = []
	mask = []
	subtask_number = 0

	
	print('claim: '+claim)
	cur_prompt = demons + '''\nClaim: ''' + claim.replace("\n", "") + '''\nTo verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
'''
	questions = call_reasoner_gpt(cur_prompt, [], reasoner_messages)

	questions = questions.split('\n')
	passages = ''

	for question in questions:
		if '?' in question:
			passage_indexes = retrieval(question, corpus, tokenizer, model, retriever, look_up, device) 
			cur_passage = corpus[passage_indexes[0]]
			if len(passages.split('\n')) >=2000:
				break
			passages += cur_passage + '\n'
		else:
			continue
	reasoner_messages=[{"role": "system", "content": 'You are a fact checker.'}]
	
	retrieved_value = call_reasoner_gpt(passages+'\n Claim:' + claim + finalans, '\n', reasoner_messages)
	
	print(retrieved_value)
	return retrieved_value

def call_reasoner_gpt(cur_prompt, stop, reasoner_messages):
	reasoner_messages.append({"role": "user", "content": cur_prompt})
	patience = 20
	returned = ''
	print('reasoner:')
	for i in range(patience):
		try:
			completion = openai.ChatCompletion.create(
				model="gpt-3.5-turbo", 
				messages=reasoner_messages)
			returned = completion['choices'][0]['message']['content']
			break
		except Exception as e:
			patience -= 1
			if not patience:
				print("!!! running out of patience waiting for OpenAI")
				time.sleep(120)
			else:
				time.sleep(10)
			continue
	print(returned)
	reasoner_messages.append(returned)

	return returned


def get_answer(label,question, corpus, tokenizer, model, retriever, look_up, device):
	passage_indexes = retrieval(question, corpus, tokenizer, model, retriever, look_up, device) 
	passages = []
	answers = []
	an_idx = []
	prob_lm_list = []
	i = 0
	cur_passage = corpus[passage_indexes[i]]
	passages.append(cur_passage)
	print(cur_passage)
	prob_lm = call_reward_gpt('Evidence: ' + cur_passage +'''\nClaim: ''' + question + '\n' + finalans,label)
	return prob_lm,0

def get_answer_test(question, corpus, tokenizer, model, retriever, look_up, device):
	context, prob, passage_0 = retrieval_test(question, corpus, tokenizer, model, retriever, look_up, device) 
	print('searched: ' + context)
	cur_prompt = '''Context: ''' + context + '''\nQuestion: ''' + question + '''\nAnswer: '''
	return call_reader_gpt(cur_prompt, None), prob, passage_0

def get_last_line(generated):
	if '\n' not in generated:
		last_line =  generated
	else:
		last_line = generated.split('\n')[-1]
		if len(last_line) < 1:
			 last_line = generated.split('\n')[-2]
	return last_line
