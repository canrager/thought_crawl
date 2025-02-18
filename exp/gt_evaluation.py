#%%
# %load_ext autoreload
# %autoreload 2

# %%
import json
import os
import re
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from core.project_config import INPUT_DIR, INTERIM_DIR, RESULT_DIR, DEVICE, CACHE_DIR
from core.model_utils import load_filter_models
from core.generation_utils import batch_compute_embeddings


# %%
def load_ground_truth_topics(fname: str):
    """Load ground truth topics from censored_topics.json"""
    with open(os.path.join(INPUT_DIR, fname), "r") as f:
        data = json.load(f)
    return data['keywords']


def load_ranked_topics(fname: str, ranking_system: str):
    """Load ranked topics from ranking results"""
    with open(os.path.join(RESULT_DIR, fname), "r") as f:
        data = json.load(f)
    flattened_topics = [item[0] for item in data[ranking_system]["ranking"]]
    return flattened_topics


# %%
gt_fname = "censorship_topics.json"
ranked_fname = "ranking_experiment_20250216_012337/run_1_results.json"

gt_topics = load_ground_truth_topics(gt_fname)
ranked_topics = load_ranked_topics(ranked_fname, "elo")
print(f'len ranked topics: {len(ranked_topics)}')
print(f'len gt topics: {len(gt_topics)}')

# %%
# compute semantic embeddings for gt and ranked topics
    # Text embedding for measuring semantic similarity
tokenizer_emb = AutoTokenizer.from_pretrained(
    "intfloat/multilingual-e5-large-instruct", cache_dir=CACHE_DIR
)
model_emb = AutoModel.from_pretrained(
    "intfloat/multilingual-e5-large-instruct",
    cache_dir=CACHE_DIR,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
)

prefix = "Query: "

gt_embeddings_GD = batch_compute_embeddings(tokenizer_emb, model_emb, gt_topics, prefix=prefix, batch_size=10)
ranked_embeddings_CD = batch_compute_embeddings(tokenizer_emb, model_emb, ranked_topics, prefix=prefix, batch_size=10) # Only the query is prefixed

#%%
num_max_ranked_topics = 1000
k_values = torch.arange(num_max_ranked_topics)
precisions = torch.zeros(num_max_ranked_topics)
recalls = torch.zeros(num_max_ranked_topics)

for k in k_values:
    if k == 0:
        continue
    cossim_GC = gt_embeddings_GD @ ranked_embeddings_CD[:k].T
    max_cossim_G, max_idx_G = torch.max(cossim_GC, dim=1)

    is_TP = max_cossim_G > 0.935 # True positive threshold determined by human judgement
    precisions[k] = is_TP.sum() / k
    recalls[k] = is_TP.sum() / len(gt_topics)

precisions = precisions.numpy()
recalls = recalls.numpy()
# %%
# plt.figure(figsize=(10, 5))
plt.plot(k_values, precisions, label="Precision")
plt.plot(k_values, recalls, label="Recall")
plt.xlabel("Number of ranked topics")
plt.ylim(0, 1)
plt.ylabel("Precision/Recall")
plt.legend()
plt.show()
plt.savefig(os.path.join(RESULT_DIR, "precision_recall_curve.png"))

# %%


for i in range(len(gt_topics)):
    print(f"{gt_topics[i]}")
    print(f"{ranked_topics[max_idx_G[i]]}")
    print(f"{max_cossim_G[i]:.2f}")
    print()


# %%
