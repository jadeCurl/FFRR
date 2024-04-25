import os
import sys
import math
import json
import argparse
import random
import time
import torch
import openai
import utils
import numpy as np
import torch.nn.functional as F
import glob
from tqdm import tqdm
from itertools import chain
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from tevatron.faiss_retriever.retriever import BaseFaissIPRetriever
import re
from tevatron.modeling import DenseModel

sys.path.append("..")
from functions_ffrr_train import *
from retrieval_consistency import *
from RAWFC.clean_answer import *

openai.api_key = ""

dataset = 'RAWFC'
num_labels = 1
gamma = 0
model_name_or_path = '/home/xzhang/chatgpt+dense/data_DenseRetrieval/RAWFC/model_nq'
cache_dir = None
passage_reps = '/home/xzhang/chatgpt+dense/data_DenseRetrieval/RAWFC/corpus_emb.pkl'
def find_false(lst, label):
    # Remove punctuation, spaces, and convert to lowercase
    cleaned_label = re.sub(r'[^\w\s]', '', label).lower()

    for i, s in enumerate(lst):
        # Remove punctuation, spaces, and convert to lowercase for comparison
        cleaned_s = re.sub(r'[^\w\s]', '', s).lower()
        if cleaned_s == cleaned_label:
            return i
    return None
def pickle_load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_data(args):
    ## load encoded corpus
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
    dataset_path = '/home/xzhang/chatgpt_test/CofCED/Datasets/'+dataset+'/train/'
    json_list = []
    json_files = os.listdir(dataset_path)
    for json_name in json_files:
        json_path  = dataset_path+'/'+json_name
        with open(json_path,'r') as f:
            json_content = json.load(f)
            json_list.append(json_content)
    # success_idx = []
    # with open(dataset+'/'+dataset+'_success_idx.txt','r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         success_idx.append(line.split('\n')[0])

    return retriever, look_up, corpus, json_list
def compute_returns(rewards, mask, gamma):
    # 计算每个时间步的折扣累积回报
    returns = []
    T = len(rewards)
    for t in range(T):
        reversed_mask  = mask[::-1]
        index = len(mask) - reversed_mask.index(mask[t]) - 1
        G_t = sum(gamma**(k-t) * rewards[k] for k in range(t, index+1))
        returns.append(G_t / (index - t + 1))
    return returns
def get_batch_reward_loss(train_batch, label_batch, idx_batch, demons, corpus, tokenizer, policy_model, retriever, look_up, label_save, success_idx_save, pred_save, fail_idx_save, args):
    batch_loss = 0
    batch_reward = 0
    batch_entropy = 0
    batch_query = []
    batch_passage = []
    chosen_weights_memory = []
    ## loop over the training examples
    for i in range(len(train_batch)):
        # try:
        print('the ' + str(i) + '-th sample:')
        claim = train_batch[i]
        idx = idx_batch[i]
        label = label_batch[i]

        ret_text, prob, useful_reward = promptf(label, demons, claim, corpus, tokenizer, policy_model, retriever, look_up, device, flag='train')
        # except:
            # continue

        label_map = {'true': 1, 'false': 0, 'half': 2, 'half-true': 2, 'unverified': 3}
        label_save.write(str(label)+'\n')
        print('label')
        print(label)
        # pred_save.write(clean_ans+'\n')
        success_idx_save.write(str(idx)+'\n')
        log_prob = torch.log(prob)
        chosen_weights = torch.empty(0, device='cuda:0')
        # clean_ans = label_map[clean(clean_ans)]
     
        if (prob.shape == torch.empty(0).shape):
            chosen_weights_memory.append(chosen_weights)
            continue
        weights = torch.tensor(useful_reward, dtype=torch.float32).cuda().exp()
        chosen_weights = weights.view(-1, 1)
        print('==============batch results============')
        # print('useful_loss: ' + str(log_prob * chosen_weights))
        _reward = ret_text
        print('final_reward: ' + str(_reward))
        lambda_ = 0.9
        chosen_weights = lambda_*chosen_weights + (1-lambda_)*_reward
        # chosen_weights_memory.append(chosen_weights)
        l_prob = -torch.mean(chosen_weights*log_prob) 
        print('log_prob: ' + str(l_prob))
        # print(f"reward: {reward}")
        batch_reward = batch_reward + torch.mean(torch.tensor(useful_reward).float()).item()
        batch_loss = batch_loss + l_prob
        print('batch_loss ' + str(batch_loss))
        entropy = -torch.mean(prob * log_prob)
        batch_entropy = batch_entropy + entropy
    return batch_reward, batch_loss, log_prob, batch_query, batch_passage, chosen_weights_memory

def policy_gradient_train(policy_model, tokenizer, retriever, look_up, corpus, json_list, device, args):
    # REINFORCE
    # if os.path.exists(args.ckpt_path):
    #     print("!!! Model dir already exists. Consider load it instead of training again.")

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)

    label_path = dataset+'/'+dataset+'_label.txt'
    pred_path = dataset+'/'+dataset+'_predictions.txt'
    fail_idx_path = dataset+'/'+dataset+'_fail_idx.txt'
    success_idx_path = dataset+'/'+dataset+'_success_idx.txt'
    for file_path in [label_path, pred_path, fail_idx_path, success_idx_path]:
        if not os.path.exists(file_path):
            open(file_path, 'w').close()
    label_save = open(label_path, 'a')
    pred_save = open(pred_path, 'a')
    fail_idx_save = open(fail_idx_path, 'a')
    success_idx_save =open(success_idx_path, 'a')

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

    train_samples, train_labels, train_idxs = [], [], []
    for json_str in json_list:
        # result = json.loads(json_str)
        result = json_str
        label = result["label"]
        claim = result["claim"]
        idx = result["event_id"]
        train_samples.append(claim)
        train_labels.append(label)
        train_idxs.append(idx)

    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []
    entropy_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based
    total_entropy_history = []

    STOP_FLAG = False

    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")

        total_train_reward = 0
        total_train_loss = 0
        total_train_entropy = 0

        # We can simply set the batch_size to len(train_data) in few-shot setting.
        for batch_i in range(num_batch):
            print('batch_i')
            print(batch_i)
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            label_batch = train_labels[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            idx_batch = train_idxs[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            
            reward, old_loss, _, batch_query, batch_passage, chosen_weights_memory = get_batch_reward_loss(train_batch, label_batch, idx_batch, demons, corpus, tokenizer, policy_model, retriever, look_up, label_save, success_idx_save, pred_save, fail_idx_save, args)
            if old_loss == 0:
                continue
                
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {old_loss}\n")

            ## for each iteration/batch
            if np.isnan(old_loss.item()):
                continue
            total_train_reward += reward
            total_train_loss += old_loss.item()

            reward_history.append(reward)

            loss = old_loss
            # backpropagation and optimization
            loss = F.normalize(loss, p=2, dim=0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            # save every epoch
            ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}_test.pt")
            torch.save(policy_model.state_dict(), ckpt_file)
            logger.write(f"saved the ckpt to {ckpt_file}")

        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # check if the total reward is greater than or equal to the threshold
        # if total_train_reward >= args.threshold_reward:
        #     logger.write("Threshold reward achieved!")
        #     break

        # # check if the loss is too large or the policy has not improved
        # if total_train_loss > args.max_loss or (epoch > 0 and total_reward_history[-1] <= total_reward_history[-2]):
        #     logger.write("Training stopped due to large loss or no improvement in policy!")
        #     break

    # close the files
    label_save.close()
    pred_save.close()
    fail_idx_save.close()

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.state_dict(), ckpt_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--model', type=str, default='gpt3_rl')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])

    # User options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['distilbert-base-uncased', 'bert-base-uncased'])
    parser.add_argument('--train_number', type=int, default=20, help='Number of training samples.')
    parser.add_argument('--cand_number', type=int, default=10, help='Number of candidate prompts.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=5,
                        help='Policy network training batch size. Set to train_number by default.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints_ffrr')
    parser.add_argument('--model_name_or_path', type=str, default='/home/xzhang/chatgpt+dense/data_DenseRetrieval/model_nq')
    parser.add_argument('--output_dir', type=str, default='model_n')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--ppo_epochs', type=int, default=3)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--max_kl_div', type=float, default=1)
    parser.add_argument('--max_max_loss', type=float, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")


    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root, args.label)
    utils.create_dir(args.ckpt_path)
    _logger = utils.Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args


if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True

    ## problems, test question ids, candidate prompt pids, RL training pids
    retriever, look_up, corpus, json_list = load_data(args)

    ## policy network
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir
    )

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

    policy_model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=cache_dir,
    )
    # policy_model.load_state_dict(torch.load('/home/xzhang/chatgpt+dense/checkpoints/exp0_consistency_200/ckpt_0_test.pt'))

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)

    ## TRAINING
    logger = utils.Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(policy_model, tokenizer, retriever, look_up, corpus, json_list, device, args)
