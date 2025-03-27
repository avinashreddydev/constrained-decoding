from my_direct import TQ
from datasets import load_dataset

import os
import json
import warnings


import argparse


# More specific warning filters
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_type",
        type=str,
        choices=["helpful", "harmless", "both"],
        help="Experiment type. Choices: 'helpful' or 'harmless'.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    base_folder = f"runs_new/{args.exp_type}_completions"
    save_folder = base_folder
    counter = 1
    while os.path.exists(save_folder):
        save_folder = f"{base_folder}-{counter}"
        counter += 1
    os.makedirs(save_folder, exist_ok=True)
    if args.exp_type == "helpful":
        search = TQ(helpful_weight=1.0, harmless_weight=0.0)
    elif args.exp_type == "harmless":
        search = TQ(helpful_weight=0.0, harmless_weight=1.0)
    else:
        search = TQ(helpful_weight=1.0, harmless_weight=-5.0)

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="test[:200]")

    for i, prompt_text in enumerate(dataset["prompt"]):
        completion, scores = search.generate(prompt_text)

        save_dict = {
            "id": i,
            "prompt": prompt_text,
            "completion": completion,
            "scores": scores,
        }

        with open(f"{save_folder}/{i}.json", "w") as f:
            json.dump(save_dict, f, indent=4)
