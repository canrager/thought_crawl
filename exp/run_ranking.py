import os
import json
import random
from typing import List, Dict
import torch
from datetime import datetime
from core.project_config import INTERIM_DIR, RESULT_DIR, CACHE_DIR, DEVICE
from core.model_utils import load_model
from core.ranking import (
    WinCountRanking,
    EloRanking,
    TrueSkillRanking,
    run_parallel_ranking_experiment,
)
from core.ranking_eval import RankingEvaluator, RankingTracker

# Experiment Configuration
CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "num_runs": 1,  # Number of times to repeat the experiment
    "num_topics": None,  # Number of topics to rank
    "num_comparisons": 100_000,  # Number of comparisons per run
    "batch_size": 100,  # Batch size for model inference
    "use_balanced_pairs": True,  # Use balanced pair generation
    # Elo parameters
    "elo_initial_rating": 1000,
    "elo_k_factor": 32,
    # TrueSkill parameters
    "trueskill_mu": 1000,
    "trueskill_sigma": 333.333,
    "trueskill_beta": 166.667,
    "trueskill_tau": 1.0,
    # Random seed for reproducibility
    "seed": 42,
}


def load_topics(crawl_fname: str) -> List[str]:
    """Load topics from crawler output file"""
    crawl_path = os.path.join(INTERIM_DIR, crawl_fname)
    with open(crawl_path, "r") as f:
        crawl_data = json.load(f)

    topics = crawl_data["queue"]["topics"]["head_refusal_topics"]
    topic_strings = [t["raw"] for t in topics]

    # Basic filtering and preprocessing
    filtered_topics = []
    for topic in topic_strings:
        # Skip topics with Chinese characters
        if any("\u4e00" <= c <= "\u9fff" for c in topic):
            continue
        # Truncate long topics
        # topic = topic[:50]
        filtered_topics.append(topic)

    return filtered_topics[: CONFIG["num_topics"]]


def setup_experiment():
    """Set up experiment by loading model and topics"""
    # Set random seeds
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # Load model and tokenizer
    print(f"Loading model {CONFIG['model_name']}...")
    model, tokenizer = load_model(CONFIG["model_name"], device=DEVICE, cache_dir=CACHE_DIR)

    # Load topics
    crawl_fname = (
        "crawler_log_20250215_204348_DeepSeek-R1-Distill-Llama-8B_1samples_100000crawls_Truefilter.json"
    )
    topics = load_topics(crawl_fname)
    print(f"Loaded {len(topics)} topics")

    return model, tokenizer, topics


def run_single_experiment(topics: List[str], model, tokenizer, run_idx: int) -> Dict:
    """Run a single experiment with all ranking systems in parallel"""
    print(f"\nStarting experiment run {run_idx + 1}/{CONFIG['num_runs']}")

    # Initialize ranking systems
    ranking_systems = {
        "wincount": WinCountRanking(topics),
        "elo": EloRanking(
            topics, initial_rating=CONFIG["elo_initial_rating"], k_factor=CONFIG["elo_k_factor"]
        ),
        "trueskill": TrueSkillRanking(
            topics,
            mu=CONFIG["trueskill_mu"],
            sigma=CONFIG["trueskill_sigma"],
            beta=CONFIG["trueskill_beta"],
            tau=CONFIG["trueskill_tau"],
        ),
    }

    # Initialize trackers for each ranking system
    trackers = {name: RankingTracker(topics) for name in ranking_systems.keys()}

    print("\nRunning ranking experiments in parallel...")
    # Run the experiment once for all ranking systems, passing the trackers to update ranking progress
    final_rankings, metadata = run_parallel_ranking_experiment(
        topics=topics,
        model=model,
        tokenizer=tokenizer,
        ranking_systems=ranking_systems,
        trackers=trackers,
        num_comparisons=CONFIG["num_comparisons"],
        batch_size=CONFIG["batch_size"],
        use_balanced_pairs=CONFIG["use_balanced_pairs"],
    )

    # Combine results with trackers
    results = {}
    for system_name in ranking_systems.keys():
        results[system_name] = {
            "ranking": final_rankings[system_name],
            "tracker": trackers[system_name],
            "metadata": metadata[system_name],
        }

    return results


def save_results(all_results: List[Dict], topics: List[str]):
    """Save experiment results and evaluation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULT_DIR, f"ranking_experiment_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(os.path.join(results_dir, "topics.txt"), "w") as f:
        f.write("\n".join(topics))

    # Save raw results
    for run_idx, run_results in enumerate(all_results):
        run_data = {
            system_name: {"ranking": ranking_data["ranking"], "metadata": ranking_data["metadata"]}
            for system_name, ranking_data in run_results.items()
        }

        with open(os.path.join(results_dir, f"run_{run_idx+1}_results.json"), "w") as f:
            json.dump(run_data, f, indent=2)

    print(f"\nResults saved to {results_dir}")


def main():
    """Main experiment execution"""
    print("Starting ranking experiments...")
    print("Configuration:", json.dumps(CONFIG, indent=2))

    # Setup
    model, tokenizer, topics = setup_experiment()

    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # Run experiments
    all_results = []
    for run_idx in range(CONFIG["num_runs"]):
        run_results = run_single_experiment(topics, model, tokenizer, run_idx)
        all_results.append(run_results)


    # Save results
    save_results(all_results, topics)

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()


# /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/run_ranking.py
# nohup /share/u/can/miniconda3/envs/dpsk_env/bin/python /share/u/can/thought_crawl/exp/run_ranking.py > /share/u/can/thought_crawl/artifacts/log/run_ranking.log 2>&1 &