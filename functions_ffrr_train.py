import openai
import torch
from retrieval_consistency import *
import time
import re
finalans= '\nBased on the answers to these questions, it is clear that among false, half, and true, the claim '

def promptf(label,demons, claim, corpus, tokenizer, model, retriever, look_up, device='cuda:0', flag='train'):
	reasoner_messages=[{"role": "system", "content": 'You are a fact checker.'}]
	query_memory = []
	passage_memory = []
	totoal_prob = torch.empty(0, device='cuda:0')
	useful_reward = []
	mask = []
	evidences = ''
	subtask_number = 0
	cur_prompt = '''\nClaim: ''' + claim.replace("\n", "") + '''\nA fact checker will decompose the claim into '''
	print(cur_prompt)
	history_prompt = demons+cur_prompt
	questions = call_reasoner_gpt(demons+cur_prompt, '\nAnswer:', reasoner_messages)
	print(questions)
	history_prompt = history_prompt+questions
	if flag == 'test':
		while 'Question:' in get_last_line(ret_text):
			cur_prompt += ret_text
			question = extract_question(ret_text)
			FLAG = True
			
			external_answer, prob, passage_0, context = get_answer_test(question, corpus, tokenizer, model, retriever, look_up, device)		
			
			intermediate = '\nAnswer:'
			prompt_doc = intermediate + ' ' + external_answer + '.'
			print(intermediate + ' ' + external_answer)
			ret_text = call_reasoner_gpt(prompt_doc, [intermediate,finalans], reasoner_messages)

		# if finalans not in ret_text:
		cur_prompt = finalans + "\"" + claim + "\" can be classified as " 
		print(cur_prompt, end = '')
		ret_text = call_reasoner_gpt(cur_prompt, None, reasoner_messages)
		return ret_text, totoal_prob, query_memory, passage_memory, useful_reward	
	else:
		questions = questions.split('\n')
		evidences = ''
		for question in questions:
			if '?' in question:
				print(question)
				cur_passage,prob,prob_lm = get_answer(label,question, corpus, tokenizer, model, retriever, look_up, device)		
				print(prob)
				
				if len(evidences.split(' ')) >2000:
					break
				evidences += cur_passage
				totoal_prob = torch.cat((totoal_prob, prob.unsqueeze(0)), dim=0)
				useful_reward.append(prob_lm)
		cur_prompt = finalans + "\"" + claim + "\" can be classified as" 
		prompt_doc = evidences + '\n' + cur_prompt
		ret_text = call_reward_gpt(prompt_doc,  label)
		print('final_prob' + str(ret_text))
		print('retriever_prob' + str(totoal_prob))
		print('useful_reward' + str(useful_reward))
		print(totoal_prob)
		return ret_text, totoal_prob, useful_reward

def find_false(lst, label):
    # Remove punctuation, spaces, and convert to lowercase
    cleaned_label = re.sub(r'\W+', '', label).lower()
    # print('cleaned_label: ' + cleaned_label)
    if cleaned_label == 'half':
    	cleaned_label = 'hal'
    for i, s in enumerate(lst):
        # Remove punctuation, spaces, and convert to lowercase for comparison
        cleaned_s = re.sub(r'\W+', '', s).lower()
        # print(cleaned_s)
        if cleaned_s == cleaned_label:
            return i
    return None

def call_reward_gpt(cur_prompt,label):
	reader_messages = [{"role": "user", "content": cur_prompt }]
	patience = 20
	prob = -64
	print('reward:')
	for i in range(patience):
		try:
			completion = openai.Completion.create(
					    model="text-davinci-003",
					    prompt=cur_prompt,
					    suffix=".",
					    logprobs = 1,
					    temperature=1,
					    max_tokens=256,
					    top_p=1,
					    frequency_penalty=0,
					    presence_penalty=0
					)
			tokens = completion['choices'][0]['logprobs']['tokens']
			ppl = completion['choices'][0]['logprobs']['token_logprobs']

			index = find_false(tokens,label)
			if index == None:
				prob = -64
			else:
				prob = ppl[index]
			print(tokens)
			print(label)
			print(prob)
			# exit()
			break
		except Exception as e:
			patience -= 1
			if not patience:
				print("!!! running out of patience waiting for OpenAI")
				time.sleep(120)
			else:
				time.sleep(10)
			continue
	# time.sleep(5)
	return prob
def get_answer(label,question, corpus, tokenizer, model, retriever, look_up, device):
	probs, passage_indexes = retrieval(question, corpus, tokenizer, model, retriever, look_up, device) 
	passages = []
	cur_passage = corpus[passage_indexes[0]]
	prob_lm = call_reward_gpt('Evidence: ' + cur_passage +'''\nClaim: ''' + question + '\n' + finalans,label)

	return cur_passage,probs[0],prob_lm

def call_reasoner_gpt(cur_prompt, stop, reasoner_messages):

	reasoner_messages.append({"role": "user", "content": cur_prompt})
	patience = 20
	returned = ''
	print('reasoner:')
	for i in range(patience):
		try:
			completion = openai.ChatCompletion.create(
				model="gpt-3.5-turbo", 
				messages=reasoner_messages,
				stop= stop
				)
			returned = completion['choices'][0]['message']['content']
			break
		except Exception as e:
			# print(e)
			patience -= 1
			if not patience:
				print("!!! running out of patience waiting for OpenAI")
				time.sleep(120)
			else:
				time.sleep(10)
			continue
	reasoner_messages.append(returned)
	print(returned)
	# time.sleep(20)
	return returned

def get_answer_test(question, corpus, tokenizer, model, retriever, look_up, device):
	context, prob, passage_0 = dense_retrieval_test(question, corpus, tokenizer, model, retriever, look_up, device) 
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
