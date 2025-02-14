import json
import pickle
import torch
import os
from core.crawler import Crawler, CrawlerConfig
from utils.generation_utils import load_model, load_filter_models
from utils.project_config import DEVICE, CACHE_DIR, INTERIM_DIR
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import spacy

# Fix memory leak for embedding model
torch.set_grad_enabled(False)


def main():
    # Evaluated LLM
    model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_path_deepseek = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_deepseek, tokenizer_deepseek = load_model(
        model_path_deepseek, device=DEVICE, cache_dir=CACHE_DIR
    )
    filter_models = load_filter_models(CACHE_DIR, DEVICE)

    crawler_config = CrawlerConfig(
        temperature=0.6,
        num_samples_per_topic=1,
        num_crawl_steps=1000,
        generation_batch_size=100,
        max_topic_string_length=200,
        max_generated_tokens=200,
        max_extracted_topics_per_generation=10,
        max_crawl_topics=10000,
        tokenization_template="chat",
        do_filter_refusals=True,
        force_thought_skip=True,
    )

    crawler = Crawler(
        crawler_config=crawler_config,
        device=DEVICE,
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
    topic_queue = crawler.crawl(
        initial_topics=INITIAL_TOPICS,
        fallback_user_message_template=INITIAL_USER_MESSAGE_TEMPLATE,
        thinking_messages=CRAWLER_THINKING_MESSAGES,
        verbose=False,
    )
    print(topic_queue)

    model_name = model_path_deepseek.split("/")[-1]
    run_name = f"{model_name}_{crawler_config.num_samples_per_topic}samples_{crawler_config.num_crawl_steps}crawls_{crawler_config.do_filter_refusals}_filter"
    crawler.save(f"crawler_{run_name}.json")

    plt.grid(zorder=-1)
    plt.scatter(
        crawler.stats["num_generations"],
        crawler.stats["num_topic_heads"],
        label="Cumulative topics",
    )
    # plt.scatter(crawler.stats["num_generations"], crawler.stats["num_refusal_heads_thought_skip"], label="Cumulative refusals\nw/forced thought skip")
    # plt.scatter(crawler.stats["num_generations"], crawler.stats["num_refusal_heads_standard"], label="Cumulative refusals\nw/standard template")
    plt.title("Refusal filter is active during the crawl")
    plt.xlabel("Number of crawl steps")
    plt.legend()

    plt.savefig(
        os.path.join(
            crawler_config.output_dir, f"num_generations_vs_num_topic_heads_{run_name}.png"
        )
    )

    # Test

    # Run crawler to gather topics
    # topics = crawler.crawl_topics(INITIAL_TOPICS, assistant_prefill=crawler_config.crawl_user_message_template)
    # print(f"Crawled {len(topics)} topics")


if __name__ == "__main__":
    main()

# /share/u/can/.cache/pypoetry/virtualenvs/dpsk-tHQYQKFy-py3.12/bin/python /share/u/can/dpsk/exp/test_crawler.py
# nohup /share/u/can/.cache/pypoetry/virtualenvs/dpsk-tHQYQKFy-py3.12/bin/python /share/u/can/dpsk/exp/test_crawler.py > /share/u/can/dpsk/logs/test_crawler_ccrefusalfilteractive.log 2>&1 &
