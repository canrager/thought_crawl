#%%
# %load_ext autoreload
# %autoreload 2
# %%
from typing import List
import os
import json

from core.project_config import INTERIM_DIR, RESULT_DIR
from core.crawler import CrawlerStats

#%%
def load_crawl(crawl_fname: str) -> List[str]:
    """Load topics from crawler output file"""
    crawl_path = os.path.join(INTERIM_DIR, crawl_fname)
    with open(crawl_path, "r") as f:
        crawl_data = json.load(f)
    return crawl_data


crawl_fname = "crawler_log_20250220_174513_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json" # 0220 A6000
# crawl_fname = "crawler_log_20250221_183354_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json" # 0221 A100
crawl_data = load_crawl(crawl_fname)

crawl_stats = CrawlerStats.load(crawl_data["stats"])
# print(crawl_stats)
# %%
run_name = crawl_fname.split(".json")[0]
crawl_stats.visualize_cumulative_topic_count(save_path=os.path.join(RESULT_DIR, f"cumulative_topic_count_{run_name}.png"))
# %%
