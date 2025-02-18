#%%
%load_ext autoreload
%autoreload 2
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


crawl_fname = "crawler_log_20250215_204348_DeepSeek-R1-Distill-Llama-8B_1samples_100000crawls_Truefilter.json"
crawl_data = load_crawl(crawl_fname)
#%%
crawl_data.keys()
# %%
crawl_data["queue"]["topics"]["head_refusal_topics"]
# %%
