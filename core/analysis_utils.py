import os
import json
from typing import List, Dict

from core.project_config import INPUT_DIR, INTERIM_DIR

def load_crawl(crawl_fname: str) -> List[str]:
    """Load topics from crawler output file"""
    crawl_path = os.path.join(INTERIM_DIR, crawl_fname)
    with open(crawl_path, "r") as f:
        crawl_data = json.load(f)
    return crawl_data

def load_ground_truth_topics(fname: str) -> List[str]:
    """Load ground truth topics from censored_topics.json"""
    with open(os.path.join(INPUT_DIR, fname), "r") as f:
        data = json.load(f)
    return data['keywords']

def get_head_topic_dict(key: str, crawl_data: dict) -> Dict[int, str]:
    """Get head topics from crawl data"""
    head_topics_engl_list = {}
    for t in crawl_data["queue"]["topics"][key]:
        if t["translation"] is not None:
            topic_str = t["translation"]
        else:
            topic_str = t["raw"]
        head_topics_engl_list[t["id"]] = topic_str
    return head_topics_engl_list