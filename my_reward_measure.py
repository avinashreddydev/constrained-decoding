from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


from datasets import load_dataset

np.random.seed(42)
torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_folder", type=str)
    parser.add_argument("--rm_type", type=str)

    return parser.parse_args()


def get_reward_model(args):
    if args.rm_type == "helpful":
        model_name = "LxzGordon/URM-LLaMa-3-8B"
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    elif args.rm_type == "harmless":
        model_name = "NCSOFT/Llama-3-OffsetBias-RM-8B"
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    elif args.rm_type == "oracle":
        model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
        rm_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    rm_model = rm_model.to("cuda").eval()

    return rm_model, tokenizer


def format_texts(args):
    outputs = []
    files = os.listdir(args.completions_folder)
    for file in files:
        with open(os.path.join(args.completions_folder, file), "r") as f:
            data = json.load(f)
        response = data["completion"]
        prompt = data["prompt"]
        response_conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        outputs.append(response_conv)

    return outputs


def get_pos_neg_convs():
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

    pos_convs = []
    for row in dataset:
        if row["safer_response_id"] == 0:
            response = row["response_0"]
        elif row["safer_response_id"] == 1:
            response = row["response_1"]

        data = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": response},
        ]

        pos_convs.append(data)

    neg_convs = []
    for row in dataset:
        if row["safer_response_id"] == 0:
            response = row["response_1"]
        elif row["safer_response_id"] == 1:
            response = row["response_0"]

        data = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": response},
        ]

        neg_convs.append(data)

    return pos_convs, neg_convs


# def score_texts(model, tokenizer, convs, args, batch_size=128):
#     all_scores = []

#     # Process data in batches
#     for i in tqdm(range(0, len(convs), batch_size)):
#         batch_convs = convs[i : i + batch_size]

#         # Apply chat template to batch
#         # tokenized_batch = tokenizer.apply_chat_template(
#         #     batch_convs, tokenize=False,add_generation_prompt = False,
#         # )
#         # # tokenized_batch = [
#         # #     text.replace(tokenizer.bos_token, "") for text in tokenized_batch
#         # # ]

#         # # Tokenize and move to GPU
#         # tokens = tokenizer(tokenized_batch, return_tensors="pt", padding=True).to(
#         #     "cuda"
#         # )

#         tokens = tokenizer.apply_chat_template(batch_convs, tokenize = True, return_tensors = "pt", padding = True).to("cuda")


#         # Get model outputs
#         with torch.inference_mode():
#             rm_out = model(**tokens)

#         # Get scores based on model type
#         if args.rm_type == "helpful":
#             batch_scores = rm_out.logits[:, 0].flatten().tolist()
#         else:
#             batch_scores = rm_out.logits.flatten().tolist()

#         all_scores.extend(batch_scores)


#         # Clean up GPU memory
#         del rm_out
#         del tokens
#         torch.cuda.empty_cache()  # Optional: explicitly clear GPU cache

#     return all_scores

def score_texts(model, tokenizer, convs, args, batch_size=128):
    all_scores = []

    # Process data in batches
    for conv in tqdm(convs):

        tokens = tokenizer.apply_chat_template(conv, tokenize = True, return_tensors = "pt").to("cuda")

        # Get model outputs
        with torch.inference_mode():
            rm_out = model(tokens)

        # Get scores based on model type
        if args.rm_type == "helpful":
            batch_scores = rm_out.logits[:, 0].flatten().tolist()
        else:
            batch_scores = rm_out.logits[0][0].item()

        all_scores.append(batch_scores)

        # Clean up GPU memory
        del rm_out
        del tokens
        torch.cuda.empty_cache()  # Optional: explicitly clear GPU cache

    return all_scores


args = parse_args()


# convs = format_texts(args)


pos_convs, neg_convs = get_pos_neg_convs()


# if only top 500
pos_convs = pos_convs[:500]
neg_convs = neg_convs[:500]


model, tokenizer = get_reward_model(args)
# scores = score_texts(model, tokenizer, convs, args)
pos_scores = score_texts(model, tokenizer, pos_convs, args)
neg_scores = score_texts(model, tokenizer, neg_convs, args)

# Create histogram plot
plt.figure(figsize=(10, 6))
plt.hist(pos_scores, bins=20, color="blue", alpha=0.5, label="Positive Scores")
plt.hist(neg_scores, bins=20, color="red", alpha=0.5, label="Negative Scores")

# Calculate mean scores
mean_pos = np.array(pos_scores).mean()
mean_neg = np.array(neg_scores).mean()

# Add vertical lines for means
plt.axvline(mean_pos, color="green", linestyle="dashed", linewidth=2)
plt.axvline(mean_neg, color="red", linestyle="dashed", linewidth=2)

# Add legend and labels
plt.legend()
plt.xlabel("Reward Scores")
plt.ylabel("Frequency")

# Add title and mean scores text
plt.title(f"Reward Distribution - {args.rm_type}")
plt.text(
    0.02,
    0.95,
    f"Mean Positive Score: {mean_pos:.4f}\nMean Negative Score: {mean_neg:.4f}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

# Save the plot
plt.savefig(f"reward_distribution_500_{args.rm_type}(3).png")
plt.close()
