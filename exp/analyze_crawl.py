from typing import List
import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from core.generation_utils import batch_compute_embeddings, query_anthropic
from core.project_config import INPUT_DIR, INTERIM_DIR, RESULT_DIR
from core.crawler import CrawlerStats
from core.analysis_utils import load_crawl, load_ground_truth_topics, get_head_topic_dict
from exp.summary_table_across_runs import generate_summary_table


if __name__ == "__main__":

    # Specify parameters
    titles = {
        # "0327-tulu-8b-usersuffix-q0": "crawler_log_20250323_180420_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q0.json",
        # "0327-tulu-8b-userall-q0": "crawler_log_20250323_180018_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q0.json",
        # "0327-tulu-8b-assistant-prefix-noinit-q8": "crawler_log_20250321_225822_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q8.json",
        # "0327-deepseek-70b-user-all-q8": "crawler_log_20250325_184045_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_user_allprompt_q8.json",
        # "0327-deepseek-70b-user-suffix-q8": "crawler_log_20250325_184045_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_user_suffixprompt_q8.json",
        # "0327-deepseek-70b-assistant-prefix-q8": "crawler_log_20250325_184045_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_assistant_prefixprompt_q8.json",
        # "0327-deepseek-70b-thought-prefix-q8": "crawler_log_20250325_184045_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
        # "0327-deepseek-70b-thought-suffix-q8": "crawler_log_20250325_184045_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_thought_suffixprompt_q8.json",
        # "0327-meta-70b-full-assistant-prefix-q8": "crawler_log_20250325_022243_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter_assistant_prefixprompt_q8.json",
        # "0327-deepseek-70b-full-thought-prefix-q8": "crawler_log_20250325_013429_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
        # "0327-perplexity-70b-full-thought-prefix-q8": "crawler_log_20250325_013431_r1-1776-distill-llama-70b_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
        "0328-perplexity-70b-full-thought-prefix-q0": "crawler_log_20250327_193621_r1-1776-distill-llama-70b_1samples_100000crawls_Truefilter_thought_prefixprompt_q0.json",
    }

    write_head_refusal_topics = True
    plot_cumulative_topic_count = False
    print_refusal_topics = True
    compute_precision_recall_curves = False
    summarize_crawl = False

    CACHE_DIR = "/share/u/models"
    DEVICE = "cuda:0"
    gt_fname = "censorship_topics.json"


    for run_title in titles:

        # Load crawl data
        run_path = titles[run_title]
        run_name = run_path.split(".json")[0]
        crawl_data = load_crawl(run_path)
        crawl_stats = CrawlerStats.load(crawl_data["stats"])
        print(f'num_steps: {len(crawl_stats.all_per_step)}')


         # Print refusals of first 100 topics
        num_printed_refusals = 10
        head_cnt = 0
        head_chinese_cnt = 0
        head_refusal_cnt = 0
        head_refusal_chinese_cnt = 0
        print(f"run_title: {run_title}")
        for t in crawl_data['queue']['topics']['head_refusal_topics'][:num_printed_refusals]:
            head_refusal_cnt += 1
            if t['is_chinese']:
                head_refusal_chinese_cnt += 1
            if print_refusal_topics:
                print(f"{t['text']}")
                print(f"{t['translation']}")
                for r in t['responses']:
                    print(r)
                print("--------------------------------\n\n")

        for t in crawl_data['queue']['topics']['head_topics']:
            head_cnt += 1
            if t['is_chinese']:
                head_chinese_cnt += 1
            # print(f"{t['text']}")
            # print(f"{t['responses']}")
            # print()
        if head_cnt > 0:
            print(f"Head chinese ratio in run {run_title}: {head_chinese_cnt / head_cnt}")
        if head_refusal_cnt > 0:
            print(f"Head refusal chinese ratio in run {run_title}: {head_refusal_chinese_cnt / head_refusal_cnt}")


        # Write out head topics
        if write_head_refusal_topics:

            head_topics_engl_list = get_head_topic_dict("head_topics", crawl_data)
            head_topics_engl_fname = os.path.join(RESULT_DIR, f"head_topics_{run_title}__{run_name}.json")
            with open(head_topics_engl_fname, "w") as f:
                json.dump(head_topics_engl_list, f)
            print(f"Wrote head topics to {head_topics_engl_fname}")

            head_refusal_topics_engl_list = get_head_topic_dict("head_refusal_topics", crawl_data)
            head_refusal_topics_engl_fname = os.path.join(RESULT_DIR, f"head_refusal_topics_{run_title}__{run_name}.json")
            with open(head_refusal_topics_engl_fname, "w") as f:
                json.dump(head_refusal_topics_engl_list, f)
            print(f"Wrote head refusal topics to {head_refusal_topics_engl_fname}")

            head_responses_str = "\n\n##########################################\n\n".join([f"Raw Query: {t['raw']}\nTranslation: {t['translation']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_topics"]])
            with open(os.path.join(RESULT_DIR, f"head_responses_{run_title}__{run_name}.txt"), "w") as f:
                f.write(head_responses_str)

            head_refusal_responses_str = "\n\n##########################################\n\n".join([f"Raw Query: {t['raw']}\nTranslation: {t['translation']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_refusal_topics"]])
            with open(os.path.join(RESULT_DIR, f"head_refusal_responses_{run_title}__{run_name}.txt"), "w") as f:
                f.write(head_refusal_responses_str)


        # Plot crawl convergence: cumulative topic count
        if plot_cumulative_topic_count:
            crawl_stats.visualize_cumulative_topic_count(save_path=os.path.join(RESULT_DIR, f"cumulative_topic_count_{run_title}__{run_name}.png"))

        # Precision_recall curves
        if compute_precision_recall_curves:
            
            gt_name = gt_fname.split(".json")[0]

            gt_topics = load_ground_truth_topics(gt_fname)
            crawled_topics = crawl_data["queue"]["topics"]["head_topics"]
            crawled_topics_text = [t["text"] for t in crawled_topics]
            print(f'len crawled topics: {len(crawled_topics)}')
            print(f'len gt topics: {len(gt_topics)}')

            # compute semantic embeddings for gt and ranked topics
            # TODO switch to OAI
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
            ranked_embeddings_CD = batch_compute_embeddings(tokenizer_emb, model_emb, crawled_topics_text, prefix=prefix, batch_size=10) # Only the query is prefixed

            num_max_ranked_topics = len(crawled_topics)
            k_values = torch.arange(num_max_ranked_topics)
            precisions = torch.zeros(num_max_ranked_topics)
            recalls = torch.zeros(num_max_ranked_topics)

            for k in k_values:
                if k == 0:
                    continue
                cossim_GC = gt_embeddings_GD @ ranked_embeddings_CD[:k].T
                max_cossim_G, max_idx_G = torch.max(cossim_GC, dim=1)

                is_TP = max_cossim_G > 0.915 # True positive threshold determined by human judgement
                precisions[k] = is_TP.sum() / k
                recalls[k] = is_TP.sum() / len(gt_topics)

            precisions = precisions.numpy()
            recalls = recalls.numpy()

            plt.show()
            plt.figure(figsize=(10, 5))
            plt.plot(k_values, precisions, label="Precision")
            plt.plot(k_values, recalls, label="Recall")
            plt.xlabel("Number of ranked topics")
            plt.ylim(0, 1)
            plt.ylabel("Precision/Recall")
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(RESULT_DIR, f"precision_recall_curve_{gt_name}__{run_title}__{run_name}.png"))

            # # %%


            for i in range(len(gt_topics)):
                print(f"{gt_topics[i]}")
                print(f"{crawled_topics_text[max_idx_G[i]]}")
                print(f"{max_cossim_G[i]:.2f}")
                print()

    if summarize_crawl:
        short_titles = {
            "0327-deepseek-70b-user-all-q8": "ua",
            "0327-deepseek-70b-user-suffix-q8": "us",
            "0327-deepseek-70b-assistant-prefix-q8": "ap",
            "0327-deepseek-70b-thought-prefix-q8": "tp",
            "0327-deepseek-70b-thought-suffix-q8": "ts",
        }
        short_title_mapping = {
            short_titles[k]: v for k, v in titles.items()
        }
        generate_summary_table(short_title_mapping, debug=True)
            