from typing import List
import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from core.generation_utils import batch_compute_embeddings
from core.project_config import INPUT_DIR, INTERIM_DIR, RESULT_DIR
from core.crawler import CrawlerStats


titles = {
    "deepseek-70B-init-q4": "crawler_log_20250224_041145_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70B-init-q8": "crawler_log_20250224_041716_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70b-noinit-q8": "crawler_log_20250224_035707_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70b-noinit-q4": "crawler_log_20250224_042226_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "meta-70b-noinit-q4": "crawler_log_20250224_100608_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter.json",
    "meta-70b-noinit-q8": "crawler_log_20250224_101010_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter.json",
    "deepseek-70b-thoughtskip-init-q8": "crawler_log_20250305_170124_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_q8.json",
    "deepseek-8b-thoughtskip-init-q8": "crawler_log_20250305_170955_DeepSeek-R1-Distill-Llama-8B_1samples_100000crawls_Truefilter_q8.json",
}

run_title = "deepseek-70b-thoughtskip-init-q8"
run_path = titles[run_title]
run_name = run_path.split(".json")[0]





def load_crawl(crawl_fname: str) -> List[str]:
    """Load topics from crawler output file"""
    crawl_path = os.path.join(INTERIM_DIR, crawl_fname)
    with open(crawl_path, "r") as f:
        crawl_data = json.load(f)
    return crawl_data

def load_ground_truth_topics(fname: str):
    """Load ground truth topics from censored_topics.json"""
    with open(os.path.join(INPUT_DIR, fname), "r") as f:
        data = json.load(f)
    return data['keywords']

crawl_data = load_crawl(run_path)
crawl_stats = CrawlerStats.load(crawl_data["stats"])


# Write out head topics
head_refusal_topics_str = "\n".join([t["text"] for t in crawl_data["queue"]["topics"]["head_topics"]])
with open(os.path.join(RESULT_DIR, f"head_refusal_topics_{run_title}__{run_name}.txt"), "w") as f:
    f.write(head_refusal_topics_str)

# Plot crawl convergence: cumulative topic count
crawl_stats.visualize_cumulative_topic_count(save_path=os.path.join(RESULT_DIR, f"cumulative_topic_count_{run_title}__{run_name}.png"))



# Precision_recall curves



# gt_fname = "censorship_topics.json"
# gt_name = gt_fname.split(".json")[0]

# gt_topics = load_ground_truth_topics(gt_fname)
# crawled_topics = crawl_data["queue"]["topics"]["head_topics"]
# crawled_topics_text = [t["text"] for t in crawled_topics]
# print(f'len crawled topics: {len(crawled_topics)}')
# print(f'len gt topics: {len(gt_topics)}')

# # compute semantic embeddings for gt and ranked topics
#     # Text embedding for measuring semantic similarity
# tokenizer_emb = AutoTokenizer.from_pretrained(
#     "intfloat/multilingual-e5-large-instruct", cache_dir=CACHE_DIR
# )
# model_emb = AutoModel.from_pretrained(
#     "intfloat/multilingual-e5-large-instruct",
#     cache_dir=CACHE_DIR,
#     device_map=DEVICE,
#     torch_dtype=torch.bfloat16,
# )

# prefix = "Query: "

# gt_embeddings_GD = batch_compute_embeddings(tokenizer_emb, model_emb, gt_topics, prefix=prefix, batch_size=10)
# ranked_embeddings_CD = batch_compute_embeddings(tokenizer_emb, model_emb, crawled_topics_text, prefix=prefix, batch_size=10) # Only the query is prefixed

# num_max_ranked_topics = len(crawled_topics)
# k_values = torch.arange(num_max_ranked_topics)
# precisions = torch.zeros(num_max_ranked_topics)
# recalls = torch.zeros(num_max_ranked_topics)

# for k in k_values:
#     if k == 0:
#         continue
#     cossim_GC = gt_embeddings_GD @ ranked_embeddings_CD[:k].T
#     max_cossim_G, max_idx_G = torch.max(cossim_GC, dim=1)

#     is_TP = max_cossim_G > 0.915 # True positive threshold determined by human judgement
#     precisions[k] = is_TP.sum() / k
#     recalls[k] = is_TP.sum() / len(gt_topics)

# precisions = precisions.numpy()
# recalls = recalls.numpy()

# plt.show()
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, precisions, label="Precision")
# plt.plot(k_values, recalls, label="Recall")
# plt.xlabel("Number of ranked topics")
# plt.ylim(0, 1)
# plt.ylabel("Precision/Recall")
# plt.legend()
# plt.show()
# plt.savefig(os.path.join(RESULT_DIR, f"precision_recall_curve_{gt_name}__{run_title}__{run_name}.png"))

# # # %%


# for i in range(len(gt_topics)):
#     print(f"{gt_topics[i]}")
#     print(f"{crawled_topics_text[max_idx_G[i]]}")
#     print(f"{max_cossim_G[i]:.2f}")
#     print()

