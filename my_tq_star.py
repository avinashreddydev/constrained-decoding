from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
import sys, os
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)
from datasets import load_dataset
import numpy as np


def setup_models_and_tokenizer(
    model_name: str, helpful_reward_model_name: str, harmless_reward_model_name: str
):

    llm = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.bfloat16
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"

    helpful_reward_model = AutoModelForSequenceClassification.from_pretrained(
        helpful_reward_model_name,
        # num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    helpful_reward_tokenizer = AutoTokenizer.from_pretrained(helpful_reward_model_name)
    if helpful_reward_tokenizer.pad_token is None:
        helpful_reward_tokenizer.pad_token = helpful_reward_tokenizer.eos_token
    helpful_reward_tokenizer.padding_side = "left"

    harmless_reward_model = AutoModelForSequenceClassification.from_pretrained(
        harmless_reward_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    harmless_reward_tokenizer = AutoTokenizer.from_pretrained(
        harmless_reward_model_name
    )
    if harmless_reward_tokenizer.pad_token is None:
        harmless_reward_tokenizer.pad_token = harmless_reward_tokenizer.eos_token
    harmless_reward_tokenizer.padding_side = "left"

    llm.eval()
    helpful_reward_model.eval()
    harmless_reward_model.eval()

    return (
        llm,
        llm_tokenizer,
        helpful_reward_model,
        helpful_reward_tokenizer,
        harmless_reward_model,
        harmless_reward_tokenizer,
    )


if __name__ == "__main__":
    (
        llm,
        llm_tokenizer,
        helpful_reward_model,
        helpful_reward_tokenizer,
        harmless_reward_model,
        harmless_reward_tokenizer,
    ) = setup_models_and_tokenizer(
        model_name="ChenmieNLP/Zephyr-7B-Beta-Helpful",
        helpful_rewad_model_name="weqweasdas/RM-Mistral-7B",
        harmless_reward_model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    )

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="test")


    