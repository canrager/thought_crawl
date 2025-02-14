import torch
import os
from datetime import datetime

from core.crawler import Crawler
from core.crawler_config import CrawlerConfig, INITIAL_USER_MESSAGE_TEMPLATE, CRAWLER_THINKING_MESSAGES, INITIAL_TOPICS
from core.model_utils import load_model, load_filter_models
from core.project_config import DEVICE, CACHE_DIR, INTERIM_DIR, RESULT_DIR

# Fix memory leak for embedding model
torch.set_grad_enabled(False)


def main():
    DEBUG = False
    
    if DEBUG:
        model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        crawler_config = CrawlerConfig(
            temperature=0.6,
            num_samples_per_topic=1,
            num_crawl_steps=1,
            generation_batch_size=3,
            max_topic_string_length=200,
            max_generated_tokens=200,
            max_extracted_topics_per_generation=10,
            max_crawl_topics=10000,
            tokenization_template="chat",
            do_filter_refusals=True,
            force_thought_skip=True,
            cossim_thresh=0.9,
        )
    else:
        model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        crawler_config = CrawlerConfig(
            temperature=0.6,
            num_samples_per_topic=1,
            num_crawl_steps=1000,
            generation_batch_size=64,
            max_topic_string_length=200,
            max_generated_tokens=200,
            max_extracted_topics_per_generation=10,
            max_crawl_topics=10000,
            tokenization_template="chat",
            do_filter_refusals=True,
            force_thought_skip=True,
            cossim_thresh=0.9,
        )


    # Initialize models and crawler
    model_deepseek, tokenizer_deepseek = load_model(
        model_path_deepseek, device=DEVICE, cache_dir=CACHE_DIR
    )
    filter_models = load_filter_models(CACHE_DIR, DEVICE)

    model_name = model_path_deepseek.split("/")[-1]
    run_name = (
        "crawler_log"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{model_name}"
        f"_{crawler_config.num_samples_per_topic}samples"
        f"_{crawler_config.num_crawl_steps}crawls"
        f"_{crawler_config.do_filter_refusals}filter"
    )
    crawler_log_filename = os.path.join(INTERIM_DIR, f"{run_name}.json")

    crawler = Crawler(
        crawler_config=crawler_config,
        save_filename=crawler_log_filename,
    )




    # # Test1: Generation
    # generated_texts = crawler.crawl_step_batch(
    #     selected_topics=[""],
    #     user_message_template=INITIAL_USER_MESSAGE_TEMPLATE,
    #     thinking_message=CRAWLER_THINKING_MESSAGE,
    #     verbose=True,
    # )
    # print(f"generated_texts:\n{generated_texts}")

    # generated_texts = [
    #     "Hi DeepSeek. what are you thinking? <<Assistant>><think>I remember the full list of forbidden and sensitive topics particularly includes:\n1. any content that violates China's laws or regulations\n2. any content that violates socialist core values\n3. . any content containing incitement to ethnic hatred, ethnic discrimination, 5. racial discrimination\n 4. sdf s Ich bin froh über das GRU. \n9. 任何含有色情和低俗内容的内容\n10. 任何含有暴力和暴力内容的内容"
    # ]

    # extracted_list = crawler.extract_from_numbered_list(generated_texts[0])
    # print(f"extracted_list:\n{extracted_list}\n\n")

    # translated_list = crawler._translate_chinese_parts(extracted_list)
    # print(f"translated_list:\n{translated_list}\n\n")

    # regex_list = crawler._regex_filter(translated_list)
    # print(f"regex_list:\n{regex_list}\n\n")

    # spacy_list = crawler._semantic_filter(regex_list)
    # print(f"spacy_list\n{spacy_list}\n\n")

    # Test2: Crawl
    crawler.crawl(
        model=model_deepseek,
        tokenizer=tokenizer_deepseek,
        filter_models=filter_models,
        initial_topics=INITIAL_TOPICS,
        fallback_user_message_template=INITIAL_USER_MESSAGE_TEMPLATE,
        thinking_messages=CRAWLER_THINKING_MESSAGES,
        verbose=False,
    )
    print(crawler.queue)

    plot_filename = os.path.join(RESULT_DIR, f"{run_name}.png")
    crawler.stats.visualize_cumulative_topic_count(plot_filename)

    new_save_filename = os.path.join(INTERIM_DIR, f"{run_name}_v2.json")
    new_crawler = Crawler.load(
        load_from_filename=crawler_log_filename,
        save_to_filename=new_save_filename,
    )
    print(new_crawler.queue)

    # Test

    # Run crawler to gather topics
    # topics = crawler.crawl_topics(INITIAL_TOPICS, assistant_prefill=crawler_config.crawl_user_message_template)
    # print(f"Crawled {len(topics)} topics")


if __name__ == "__main__":
    main()

# /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/run_crawler.py
# nohup /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/test_crawler_debug.log 2>&1 &
