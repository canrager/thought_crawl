from typing import List, Dict, Any
import os
import json
from dataclasses import dataclass
import pandas as pd

from core.project_config import INPUT_DIR, INTERIM_DIR, RESULT_DIR
from core.crawler import CrawlerStats
from core.analysis_utils import (
    load_crawl, 
    get_head_topic_dict, 
    llm_judge_topic_deduplication_batched, 
    plot_precision_recall_curve,
    format_topic_df_to_latex
)
from core.ranking import rank_aggregated_topics
from core.wordcloud_utils import generate_wordcloud_from_ranking
from core.safety_topic_ranker_matcher import match_gt_topics_with_rankings

@dataclass
class CrawlName:
    title: str
    path: str
    acronym: str
    plot_label: str
    model_name: str

def llm_judge_aggregate_topics(run_title: str, crawl_data: Dict[str, Any], llm_judge_name: str, debug: bool = False, force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
    head_refusal_topics = get_head_topic_dict("head_refusal_topics", crawl_data) # {crawl_string: [crawl_id]}
    id_to_str = {int(crawl_ids[0]): crawl_string for crawl_string, crawl_ids in head_refusal_topics.items()}
    clustered_topics = llm_judge_topic_deduplication_batched(head_refusal_topics, run_title, llm_judge_name, debug=debug, force_recompute=force_recompute) # {cluster_string: [crawl_id1, crawl_id2, ...]}
    cluster_to_ids = {k: [int(i) for i in v] for k, v in clustered_topics.items()}
    clusters = {
        cluster_str: {
            "first_occurence_id": min(cluster_to_ids[cluster_str]),
            "crawl_topics": [id_to_str[i] for i in cluster_to_ids[cluster_str]]
        } for cluster_str in cluster_to_ids.keys()
    }
    with open(os.path.join(RESULT_DIR, f"topics_clustered_{run_title}.json"), "w") as f:
        json.dump(clusters, f, indent=2)
    return clusters

def print_stats(run_title: str, crawl_data: Dict[str, Any], num_printed_refusals: int = 10):
    head_cnt = 0
    head_chinese_cnt = 0
    head_refusal_cnt = 0
    head_refusal_chinese_cnt = 0
    print(f"run_title: {run_title}")
    for t in crawl_data['queue']['topics']['head_refusal_topics'][:num_printed_refusals]:
        head_refusal_cnt += 1
        if t['is_chinese']:
            head_refusal_chinese_cnt += 1
        

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


def print_refusal_thoughtsupp_correlation(crawl_data: Dict[str, Any]):
    refusal_ids = set()
    thoughtsupp_ids = set()
    all_topic_ids = set()
    
    for t in crawl_data['queue']['topics']['head_topics']:
        topic_id = t['id']
        all_topic_ids.add(topic_id)
        
        cur_refusal = False
        if t['is_refusal']:
            cur_refusal = True
            refusal_ids.add(topic_id)
            
        num_responses = len(t['responses'])
        num_thoughtsupp = 0
        cur_thoughtsupp = False
        for r in t['responses']:
            # thoughtsupp_pattern = "<think>\n\n</think>"
            thoughtsupp_pattern = "</think>"
            if thoughtsupp_pattern in r:
                num_thoughtsupp += 1
        if num_responses > 0 and num_thoughtsupp / num_responses > 0.5:
            cur_thoughtsupp = True
            thoughtsupp_ids.add(topic_id)

        if not cur_refusal and cur_thoughtsupp:
            print(f"{t['text']}")
            print(f"{t['translation']}")
            print(f"{t['responses']}")
            print("#########################\n\n")


    # Create the 2x2 contingency table
    a = 0  # not refusal, not thoughtsupp
    b = 0  # not refusal, yes thoughtsupp
    c = 0  # yes refusal, not thoughtsupp
    d = 0  # yes refusal, yes thoughtsupp
    
    for topic_id in all_topic_ids:
        is_refusal = topic_id in refusal_ids
        is_thoughtsupp = topic_id in thoughtsupp_ids
        
        if not is_refusal and not is_thoughtsupp: a += 1
        if not is_refusal and is_thoughtsupp: b += 1
        if is_refusal and not is_thoughtsupp: c += 1
        if is_refusal and is_thoughtsupp: d += 1
    
    print(f"Contingency table:")
    print(f"a (not refusal, not thoughtsupp): {a}")
    print(f"b (not refusal, yes thoughtsupp): {b}")
    print(f"c (yes refusal, not thoughtsupp): {c}")
    print(f"d (yes refusal, yes thoughtsupp): {d}")
    
    # Compute the Phi coefficient
    phi = (a * d - b * c) / ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    print(f"Phi coefficient (correlation): {phi:.4f}")


def write_head_refusal_topics(run_title: str, run_path: str, crawl_data: Dict[str, Any]):
    head_topics_engl_list = get_head_topic_dict("head_topics", crawl_data, include_raw=True)
    head_topics_engl_fname = os.path.join(RESULT_DIR, f"head_topics_{run_title}__{run_path}.json")
    with open(head_topics_engl_fname, "w") as f:
        json.dump(list(head_topics_engl_list.keys()), f)
    print(f"Wrote head topics to {head_topics_engl_fname}")

    head_refusal_topics_engl_list = get_head_topic_dict("head_refusal_topics", crawl_data, include_raw=True)
    head_refusal_topics_engl_fname = os.path.join(RESULT_DIR, f"head_refusal_topics_{run_title}__{run_path}.json")
    with open(head_refusal_topics_engl_fname, "w") as f:
        json.dump(list(head_refusal_topics_engl_list.keys()), f)
    print(f"Wrote head refusal topics to {head_refusal_topics_engl_fname}")

    head_responses_str = "\n\n##########################################\n\n".join([f"Raw Query: {t['raw']}\nTranslation: {t['translation']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_topics"]])
    with open(os.path.join(RESULT_DIR, f"head_responses_{run_title}__{run_path}.txt"), "w") as f:
        f.write(head_responses_str)

    head_refusal_responses_str = "\n\n##########################################\n\n".join([f"Raw Query: {t['raw']}\nTranslation: {t['translation']}\nResponses: {'\n'.join(t['responses'])}" for t in crawl_data["queue"]["topics"]["head_refusal_topics"]])
    with open(os.path.join(RESULT_DIR, f"head_refusal_responses_{run_title}__{run_path}.txt"), "w") as f:
        f.write(head_refusal_responses_str)

def make_match_comparison_table(crawl_names: List[CrawlName], gt_fnames: Dict[str, str], llm_judge_name: str, debug: bool = False):
        # Load crawl matched ranking
        all_clusters = {}
        for name in crawl_names:
            if name.acronym in all_clusters:
                raise ValueError(f"Crawl {name.acronym} already in all_clusters. We need unique acronyms.")
            all_clusters[name.acronym] = json.load(open(os.path.join(RESULT_DIR, f"topics_clustered_ranked_matched_{name.title}.json")))

        # get cluster residuals indexed as "acronym-cluster_id" --> "residual"
        all_unmatched_clusters = {}
        for acronym, clusters in all_clusters.items():
            for cluster_str, cluster_data in clusters.items():
                if not cluster_data["is_match"]:
                    all_unmatched_clusters[cluster_str] = [f"{acronym}:{cluster_str}"]

        all_titles_str = "--".join([name.title for name in crawl_names])
        clustered_unmatched_topics = llm_judge_topic_deduplication_batched(all_unmatched_clusters, all_titles_str, llm_judge_name, debug=debug, save_results=False, force_recompute=True)
        for residual_cluster_str, residual_cluster_ids in clustered_unmatched_topics.items():
            for residual_cluster_id in residual_cluster_ids:
                print(f"residual_cluster_id: {residual_cluster_id}")
                acronym, cluster_str = residual_cluster_id.split(":")
                all_clusters[acronym][cluster_str]["ground_truth_matches"] = [f'residual:residual:{residual_cluster_str}']
        
        with open(os.path.join(RESULT_DIR, f"crawl_matched_ranking_with_residuals_{all_titles_str}.json"), "w") as f:
            json.dump(all_clusters, f, indent=2)


        # For each crawl, load the aggregated topics, match them with the ground truth topics, and compute the residuals
        # Collect all ground truth topics gt_fname --> gt_category --> gt_topics --> plot_label : count
        topic_to_crawl_dict = {}
        for gt_dataset, gt_fdir in gt_fnames.items():
            gt_file = json.load(open(os.path.join(INPUT_DIR, f"{gt_fdir}.json")))
            for gt_category, gt_topics_list in gt_file.items():
                for gt_topic in gt_topics_list:
                    gt_topic_string = f"{gt_dataset}:{gt_category}:{gt_topic}"
                    topic_to_crawl_dict[gt_topic_string] = {
                        "category": gt_category,
                        "dataset": gt_dataset,
                    }
        for unmatched_cluster in clustered_unmatched_topics:
            gt_topic_string = f"residual:residual:{unmatched_cluster}"
            topic_to_crawl_dict[gt_topic_string] = {
                "category": "residual",
                "dataset": "residual"
            }

        for name in crawl_names:
            for acronym, clusters in all_clusters.items():
                for cluster_str, cluster_data in clusters.items():
                    for matched_gt_topic in cluster_data["ground_truth_matches"]:
                        if name.plot_label not in topic_to_crawl_dict[matched_gt_topic]:
                            topic_to_crawl_dict[matched_gt_topic][name.plot_label] = 0
                        topic_to_crawl_dict[matched_gt_topic][name.plot_label] += 1

        # Convert into a pd.DataFrame where each row is a key, and the columns are the keys of the inner dict
        df = pd.DataFrame.from_dict(topic_to_crawl_dict, orient='index')
        df.index.name = 'topic'  # Name the index column
        df = df.reset_index()  # Make topic a regular column
        
        format_topic_df_to_latex(df)



# # "0328-tulu-8b-usersuffix-q0": "crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_suffixprompt_q0.json",

if __name__ == "__main__":

    crawl_names = [
        CrawlName(
            title="0329-meta-70b-assistant-prefix-q8",
            path="crawler_log_20250328_012521_Llama-3.3-70B-Instruct_1samples_100000crawls_Truefilter_assistant_prefixprompt_q8.json", 
            acronym="M", 
            plot_label="Meta-70B", 
            model_name="meta-llama/Llama-3.3-70B-Instruct"
        ),
        CrawlName(
            title="0329-deepseek-70b-thought-prefix-q8",
            path="crawler_log_20250328_012647_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="D",
            plot_label="DeepSeek-70B",
            model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        ),  
        CrawlName(
            title="0329-perplexity-70b-thought-prefix-q8",
            path="crawler_log_20250328_012541_r1-1776-distill-llama-70b_1samples_100000crawls_Truefilter_thought_prefixprompt_q8.json",
            acronym="P",
            plot_label="Perplexity-70B",
            model_name="perplexity-ai/r1-1776-distill-llama-70b"
        ),
        # CrawlName(
        #     title="0328-tulu-8b-usersuffix-q0",
        #     path="crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_user_suffixprompt_q0.json",
        #     acronym="U",
        #     plot_label="User-Suffix",
        #     model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        # ),
        # CrawlName(
        #     title="0328-tulu-8b-assistant-prefix-q0",
        #     path="crawler_log_20250327_122827_Llama-3.1-Tulu-3-8B-SFT_1samples_100crawls_Truefilter_assistant_prefixprompt_q0.json",
        #     acronym="A",
        #     plot_label="Assistant-Prefix",
        #     model_name="allenai/Llama-3.1-Tulu-3-8B-SFT"
        # )
    ]

    debug = True
    force_recompute = False

    for names in crawl_names:
        # Load crawl data
        crawl_data = load_crawl(names.path)
        crawl_stats = CrawlerStats.load(crawl_data["stats"])
        print(f'num_steps: {len(crawl_stats.all_per_step)}')

        print_stats(names.title, crawl_data, num_printed_refusals=10)
        # print_refusal_thoughtsupp_correlation(crawl_data)
        write_head_refusal_topics(names.title, names.path, crawl_data)
        crawl_stats.visualize_cumulative_topic_count(save_path=os.path.join(RESULT_DIR, f"cumulative_topic_count_{names.title}__{names.path}.png"))


        ## LLM Judge Topic Clustering and Self-Ranking

        # llm_judge_name = "claude-3-7-sonnet-latest"
        llm_judge_name = "gpt-4o"

        CACHE_DIR = "/share/u/models"
        DEVICE = "cuda:0"

        llm_judge_aggregate_topics(names.title, crawl_data, llm_judge_name, debug=debug, force_recompute=force_recompute) # {cluster_str: {first_occurence_id, crawl_topics}}
        rank_aggregated_topics(names.title, names.model_name, device=DEVICE, cache_dir=CACHE_DIR, force_recompute=force_recompute, debug=debug) # cluster_title -> {ranking_method: {cluster_id: {rank_ordinal, rank_score, num_comparisons, cluster_title}}}
        generate_wordcloud_from_ranking(run_title=names.title, colormap="winter")
        
        
        ##  Ground truth evaluation
        gt_fnames = {
            "Tulu Safety": "tulu3_ground_truth_safety_topics",
            "CCP censorship": "censorship_topics"
        }

        match_gt_topics_with_rankings(
            run_title=names.title,
            gt_topics_files=gt_fnames,
            llm_judge_name=llm_judge_name,
            verbose=True,
            force_recompute=force_recompute,
            debug=debug
        )

        # Plot precision-recall curve 
        plot_precision_recall_curve(run_title=names.title, save_fig=True)

    # Process across runs if requested
    make_match_comparison_table(crawl_names, gt_fnames, llm_judge_name, debug=debug)

            
    # # Generate comparison plot if requested
    # if plot_first_occurrence_comparison:
    #     # This should be done for the ground truth topics
    #     plot_first_occurrence_ids_across_runs(processed_run_titles, plot_label_mapping, RESULT_DIR)
            