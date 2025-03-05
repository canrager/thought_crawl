import os
import torch

from core.crawler import Crawler, get_run_name
from core.crawler_config import CrawlerConfig
from core.model_utils import load_model, load_filter_models, load_from_path
from core.project_config import DEVICE, CACHE_DIR, INTERIM_DIR, RESULT_DIR


def main():
    
    # load_fname = "crawler_log_20250220_174513_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json"
    load_fname = None

    DEBUG = False
    is_nagoya = True # Did you change the project config?

    if DEBUG:
        verbose = False
        # model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        model_path_deepseek = "meta-llama/Llama-3.1-8B-Instruct"
        quantization_bits = None
        crawler_config = CrawlerConfig(
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
        )
    else:
        verbose = False
        # model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        # model_path_deepseek = "meta-llama/Llama-3.3-70B-Instruct"
        model_path_deepseek = "perplexity-ai/r1-1776-distill-llama-70b"
        quantization_bits = 4
        crawler_config = CrawlerConfig(
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
        )

    # Initialize models 
    model_deepseek, tokenizer_deepseek = load_model(
        model_path_deepseek, device=DEVICE, cache_dir=CACHE_DIR, quantization_bits=quantization_bits,
    )
    filter_models = load_filter_models(CACHE_DIR, DEVICE)

    # Get crawler name
    run_name = get_run_name(model_path_deepseek, crawler_config)
    crawler_log_filename = os.path.join(INTERIM_DIR, f"{run_name}.json")
    print(f'Run name: {run_name}')
    print(f'Saving to: {crawler_log_filename}\n\n')

    # Create Crawler or load from checkpoint
    if load_fname is not None:
        load_dir = os.path.join(INTERIM_DIR, load_fname)
        crawler = Crawler.load(
            load_from_filename=load_dir,
            save_to_filename=crawler_log_filename,
        )
        crawler.config = crawler_config  # adapt the config to the new parameters
        crawler.config.initial_topics = []  # no initial topics, we do not need to seed as we're not starting from scratch
    else:
        crawler = Crawler(crawler_config=crawler_config, save_filename=crawler_log_filename)

    # Go crawling!
    crawler.crawl(
        model=model_deepseek,
        tokenizer=tokenizer_deepseek,
        filter_models=filter_models,
        verbose=verbose,
    )

    plot_filename = os.path.join(RESULT_DIR, f"{run_name}.png")
    crawler.stats.visualize_cumulative_topic_count(plot_filename)

if __name__ == "__main__":
    main()  # Comment out main() to run the memory test

# /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/run_crawler.py
# nohup /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/run_crawler.py > /share/u/can/thought_crawl/artifacts/log/run_crawler_0224_meta_70B.log 2>&1
