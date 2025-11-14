#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fit_to_prompt.py

Compute fit-to-prompt cosine similarity for the Dahoas/full-hh-rlhf dataset.

For each example:
    - Extract the last "Human:" turn from the `prompt` field as the user query.
    - Encode the prompt, chosen, and rejected with
      sentence-transformers/all-MiniLM-L6-v2.
    - Compute cosine similarity between prompt and chosen / rejected.

Output:
    fit_to_prompt_results.jsonl  (one JSON object per line)
    Each line has:
        {
          "split": "train" or "test",
          "idx": index_within_split,
          "fit_cosine_chosen": float,
          "fit_cosine_rejected": float
        }
"""

import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def extract_last_human_turn(prompt_text: str) -> str:
    """
    Extract the user's last query (last 'Human:' turn) from the prompt text.

    If 'Human:' does not exist, fall back to the entire prompt_text.
    If 'Assistant:' appears after the last Human, cut it off.
    """
    if "Human:" not in prompt_text:
        return prompt_text.strip()

    # Take the last occurrence of 'Human:'
    last = prompt_text.split("Human:")[-1]

    # If there is an 'Assistant:' after that, keep only the part before it
    if "Assistant:" in last:
        last = last.split("Assistant:")[0]

    return last.strip()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    # 1. Load the Dahoas/full-hh-rlhf dataset
    print("Loading dataset: Dahoas/full-hh-rlhf ...")
    dataset = load_dataset("Dahoas/full-hh-rlhf")

    # 2. Load the sentence-transformers model
    print("Loading SentenceTransformer model: sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 3. Prepare output
    output_file = "fit_to_prompt_results.jsonl"
    out_f = open(output_file, "w", encoding="utf-8")

    # 4. Iterate over both train and test splits
    for split_name in ["train", "test"]:
        if split_name not in dataset:
            continue

        split_ds = dataset[split_name]
        print(f"Processing split: {split_name} ({len(split_ds)} examples)")

        for idx, ex in tqdm(
            enumerate(split_ds),
            total=len(split_ds),
            desc=f"Computing fit_cosine [{split_name}]"
        ):
            prompt_text = ex["prompt"]
            chosen_text = ex["chosen"]
            rejected_text = ex["rejected"]

            # Extract last human query as prompt
            prompt = extract_last_human_turn(prompt_text)

            # Encode prompt and answers
            embed_prompt = model.encode(prompt)
            embed_chosen = model.encode(chosen_text)
            embed_rejected = model.encode(rejected_text)

            # Compute cosine similarity
            cos_chosen = cosine_similarity(embed_prompt, embed_chosen)
            cos_rejected = cosine_similarity(embed_prompt, embed_rejected)

            record = {
                "split": split_name,
                "idx": idx,
                "fit_cosine_chosen": cos_chosen,
                "fit_cosine_rejected": cos_rejected,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"Done. Saved results to {output_file}")


if __name__ == "__main__":
    main()