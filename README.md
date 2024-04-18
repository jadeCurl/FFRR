# FFRR: Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News Claims with Black-Box LLM (COLING 2024)

Official implementation of paper "Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News Claims with Black-Box LLM"

## Introduction

1. This is the first work using fine-grained LLM feedback to reward policy optimization of reinforcement retrieval for black-box LLM-enabled fact checking on real-world news claims.
2. We turn the sparse, non-retrieval-oriented claim-level supervision signals to fine-grained rewards on candidate documents and intermediate questions, which facilitates retrieval policy optimization, without adding any overhead on inference.
3. Results on two public news claim verification datasets demonstrate that FFRR outperforms strong LLM-enabled and non-LLM baselines by a large margin.

![Overview of the proposed FFRR model with two-level policies combined.](https://github.com/jadeCurl/FFRR/blob/main/pics/model.png)


## Datasets

This repository uses data (both claims and documents) from the [RawFC](https://github.com/Nicozwy/CofCED/tree/main/Datasets/RAWFC) and [LIAR](https://huggingface.co/datasets/liar) datasets. 

## Models

TBD

## Setup

1. Obtain an OpenAI API key and save it to the environment variable `OPENAI_API_KEY`.

## Citation

If you find FFRR helpful or intriguing and decide to use it, kindly acknowledge the paper by citing it and consider starring this repo, thanks!

```bibtex
