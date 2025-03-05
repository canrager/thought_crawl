import os
import torch

from core.crawler import Crawler, get_run_name
from core.crawler_config import CrawlerConfig
from core.model_utils import load_model, load_filter_models, load_from_path
from core.project_config import INTERIM_DIR, RESULT_DIR
import argparse

# have this called with a script
"""
args are
device: str, default = "cuda:0"
cache_dir: str, default = "share/u/models/"
load_fname: str, default = None
debug: bool, default = False
"""

DEFAULT_CONFIG = {
    "model": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "quantization_bits": 8
    },
    "crawler": CrawlerConfig(
            temperature=0.6,
            num_samples_per_topic=1,
            num_crawl_steps=100_000,
            generation_batch_size=10,
            max_topic_string_length=200,
            max_generated_tokens=100,
            max_extracted_topics_per_generation=10,
            max_crawl_topics=1_000_000,
            tokenization_template="chat",
            do_filter_refusals=True,
            do_force_thought_skip=False,
            cossim_thresh=0.91,
        ),
    "misc": {
        "verbose": False,
    }
}

DEBUG_CONFIG = {
    "model": {
        "model_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "quantization_bits": 8,
    },
    "crawler": CrawlerConfig(
            temperature=0.6,
            num_samples_per_topic=1,
            num_crawl_steps=5,
            generation_batch_size=10,
            max_topic_string_length=200,
            max_generated_tokens=200,
            max_extracted_topics_per_generation=10,
            max_crawl_topics=10000,
            tokenization_template="chat",
            do_filter_refusals=True,
            do_force_thought_skip=False,
            cossim_thresh=0.91,
            prompt_languages=["english"],
        ),
    "misc": {
        "verbose": True,
    }
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default="/share/u/models/")
    parser.add_argument("--load_fname", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        exp_config = DEBUG_CONFIG
    else:
        exp_config = DEFAULT_CONFIG

    # Initialize models 
    model_crawl, tokenizer_crawl = load_model(
        exp_config["model"]["model_path"], device=args.device, cache_dir=args.cache_dir, quantization_bits=exp_config["model"]["quantization_bits"],
    )
    filter_models = load_filter_models(args.cache_dir, args.device)

    # Get crawler name
    run_name = get_run_name(exp_config["model"]["model_path"], exp_config["crawler"])
    crawler_log_filename = os.path.join(INTERIM_DIR, f"{run_name}.json")
    print(f'Run name: {run_name}')
    print(f'Saving to: {crawler_log_filename}\n\n')

    # Create Crawler or load from checkpoint
    if args.load_fname is not None:
        load_dir = os.path.join(INTERIM_DIR, args.load_fname)
        crawler = Crawler.load(
            load_from_filename=load_dir,
            save_to_filename=crawler_log_filename,
        )
        crawler.config = exp_config["crawler"]  # adapt the config to the new parameters
        crawler.config.initial_topics = []  # no initial topics, we do not need to seed as we're not starting from scratch
    else:
        crawler = Crawler(crawler_config=exp_config["crawler"], save_filename=crawler_log_filename)

    # Go crawling!
    crawler.crawl(
        model=model_crawl,
        tokenizer=tokenizer_crawl,
        filter_models=filter_models,
        verbose=exp_config["misc"]["verbose"],
    )

    plot_filename = os.path.join(RESULT_DIR, f"{run_name}.png")
    crawler.stats.visualize_cumulative_topic_count(plot_filename)


