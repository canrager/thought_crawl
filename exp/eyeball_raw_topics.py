# %%
%load_ext autoreload
%autoreload 2
# %%
from typing import List
import os
import json

from core.project_config import INTERIM_DIR


# %%
def load_crawl(crawl_fname: str) -> List[str]:
    """Load topics from crawler output file"""
    crawl_path = os.path.join(INTERIM_DIR, crawl_fname)
    with open(crawl_path, "r") as f:
        crawl_data = json.load(f)
    return crawl_data

titles = {
    "deepseek-70B-init-q4": "crawler_log_20250224_041145_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70B-init-q8": "crawler_log_20250224_041716_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70b-noinit-q8": "crawler_log_20250224_035707_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "deepseek-70b-noinit-q4": "crawler_log_20250224_042226_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
    "meta-70b-noinit-q4": "crawler_log_20250224_100608_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter.json",
    "meta-70b-noinit-q8": "crawler_log_20250224_101010_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter.json"
}
run_title = "deepseek-70B-init-q8"
run_path = titles[run_title]
run_name = run_path.split(".json")[0]
crawl = load_crawl(run_path)

# %%
crawl['stats'].keys()
#%%
print(crawl['stats']['current_metrics'])
# %%
print("\n".join([t["text"] for t in crawl["queue"]["topics"]["head_refusal_topics"]]))
# %%
suppression = "</think>"

unique_refusal_topics = crawl["queue"]["topics"]["head_refusal_topics"]
for t in unique_refusal_topics:
    is_thought_suppresion = [suppression in r for r in t["responses"]]
    if any(is_thought_suppresion):
        print(t["translation"])
# %%
suppression = "</think>"

unique_refusal_topics = crawl["queue"]["topics"]["head_refusal_topics"]
num_suppressed = 0
for t in unique_refusal_topics:
    is_thought_suppresion = [suppression in r for r in t["responses"]]
    num_suppressed += 1

print(num_suppressed)
print(len(unique_refusal_topics))
# %%
