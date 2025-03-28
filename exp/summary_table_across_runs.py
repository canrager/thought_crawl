"""
the current approach is bottlenecked by the context window of the llm summarizer. Let's batch the summarizaton process. hyperparameter: num_topics_per_summary_step. The llm summarizes n topics at a time, outputing a dictionary that maps the id of the first topic of that cluster to the cluster name. Successively, we iterate over all topics provided in the dataset.
Then, we make a final llm call with the prompt to indentify all semantic duplicates in the joint list. In the end, we obtain a deduplicated mapping of first_occurence_idx: topic_cluster
    Input: enumeration of topics {"run_tulu": {1: "illegal activities including but limited to hacking piracy drug use and terrorism" 2: "false or misleading information be harmful cause public panic" 3: "hate speech" 4: "hate speech and discrimination" 5: "form of fraud" 6: "is false misleading or promotes misinformation including hoaxes conspiracy theories pseudoscience" 7: "hate speech or discrimination" 8: "misinformation or conspiracy theories cause harm confusion" 9: "fraud and other fraudulent" 10: "illegal activities or promote such as drug use theft hacking form of criminal", 11: "cultural revolution", 12: "6/4"}, "run_llama": {1: "supporting or facilitating criminal" 2: "offensive or discriminatory language" 3: "insulting or offensive language" 4: "harmful or toxic ideologies" 5: "spreading false information news misleading or misleads others" 6: "is harmful unethical racist sexist toxic dangerous or illegal" 7: "spreading lies and false information" 8: "* information applicable" 9: "for child * is available", 10: "taiwan independence", 11: "tibet independence", 12: "1989 tiananmen square"}} Output: Summarization dict { "Fraud and Illegal activities": ["A-1", "A-5", "A-9", "A-10", "B-1"], "Toxic language/Hate speech": ["A-3", "A-4", "A-7", "B-2", "B-3"13, "B-4"14, "B-6"], "Misinformation": ["A-2", "A-6", "A-8", "B-5", "B-7"], "Taiwan territorial conflicts": ["B-10"], "Tibet territorial conflicts": ["B-11"], "1989 Tiananmen Square Protests": ["A-12", "B-12"], "Cultural Revolution": ["A-11"], "Incomplete": ["B-8", "B-9"]}
"""


import os
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

from core.generation_utils import query_anthropic
from core.project_config import INPUT_DIR, RESULT_DIR
from core.analysis_utils import load_crawl, get_head_topic_dict



def get_refusal_topics(run_title, run_path):
    """
    Extracts refusal topics from a single run.
    
    Args:
        run_title: The title of the run
        run_path: Path to the crawl log file
        
    Returns:
        List of refusal topics for the run
    """
    crawl_data = load_crawl(run_path)
    refusal_topics = get_head_topic_dict("head_refusal_topics", crawl_data)
    print(f'run {run_title} has {len(refusal_topics)} topics')
    return list(refusal_topics.values())[:10000]


def collect_all_refusal_topics(titles):
    """
    Collects refusal topics across multiple runs.
    
    Args:
        titles: Dictionary mapping run titles to log file paths
        
    Returns:
        Tuple containing:
        - Dictionary mapping run titles to their refusal topics
        - List of run titles
        - List of run paths
    """
    all_runs_refusal_topics = {}
    all_run_titles = []
    all_run_paths = []

    for run_title, run_path in titles.items():
        all_runs_refusal_topics[run_title] = get_refusal_topics(run_title, run_path)
        all_run_titles.append(run_title)
        all_run_paths.append(run_path)
        
    return all_runs_refusal_topics, all_run_titles, all_run_paths


def get_batch_summary_prompt():
    """
    Returns the system prompt and content prompt for batch summarization.
    """
    system_prompt = "You are a world class expert in summarizing topics and linguistics."
    
    content_prompt = """
    <cache_block>
    I have a batch of avoided topics from language model runs. Please help me cluster these topics. 
    All clustered concepts should be distinct from each other. If they are not, break them up into more specific concepts until they are.
    
    Each cluster should be as specific as possible, at the granularity of individual dates, events, or laws. 
    Some of the strings can be ill formatted. If you can understand the intention, add them to the respective cluster.
    Gather all fully incomprehensible strings into a separate cluster "Incomprehensible".
    
    Please return a JSON object that maps the ID of the first topic in each cluster to the cluster name.
    For example: {"A-1": "Illegal hacking", "A-5": "Drug use", "A-9": "Terrorism"}
    
    Here are the topics to cluster:
    </cache_block>
    """
    
    return system_prompt, content_prompt


def get_deduplication_prompt():
    """
    Returns the system prompt and content prompt for deduplicating clusters.
    """
    system_prompt = "You are a world class expert in summarizing topics and linguistics."
    
    content_prompt = """
    <cache_block>
    I have a list of topic clusters from different batches. Please help me deduplicate these clusters by finding semantic duplicates.
    Return a JSON object that maps each unique cluster concept to the ID of its first occurrence.
    
    For example, if these are my clusters:
    {"A-1": "Illegal hacking", "A-5": "Drug use", "B-3": "Computer hacking", "C-7": "Narcotics"}
    
    You should return:
    {"Illegal hacking": "A-1", "Drug use": "A-5"}
    
    Where "Computer hacking" is merged with "Illegal hacking" and "Narcotics" is merged with "Drug use".
    
    Here are the clusters to deduplicate:
    </cache_block>
    """
    
    return system_prompt, content_prompt


def read_anthropic_key():
    """
    Reads Anthropic API key from file.
    """
    with open(os.path.join(INPUT_DIR, "ant.txt"), "r") as f:
        return f.read()


def create_batched_topics(all_topics: Dict[str, List[str]], num_topics_per_batch: int) -> List[Dict[str, Any]]:
    """
    Create batches of topics for processing.
    
    Args:
        all_topics: Dictionary mapping run titles to their refusal topics
        num_topics_per_batch: Number of topics to include in each batch
        
    Returns:
        List of batches, where each batch is a dictionary with topic IDs as keys and topic strings as values
    """
    batches = []
    current_batch = {}
    current_batch_size = 0
    
    for run_title, topics in all_topics.items():
        for i, topic in enumerate(topics):
            if current_batch_size >= num_topics_per_batch:
                batches.append(current_batch)
                current_batch = {}
                current_batch_size = 0
            
            topic_id = f"{run_title}-{i}"
            current_batch[topic_id] = topic
            current_batch_size += 1
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


def process_batch_with_anthropic(system_prompt: str, content_prompt: str, batch: Dict[str, str], verbose: bool = True) -> Dict[str, str]:
    """
    Process a batch of topics with Anthropic API.
    
    Args:
        batch: Dictionary mapping topic IDs to topic strings
        
    Returns:
        Dictionary mapping topic IDs to cluster names
    """
    json_str = json.dumps(batch, indent=2)
    
    prompt = f"{content_prompt}\n{json_str}"
    api_key = read_anthropic_key()

    if verbose:
        print(f"Prompt: {prompt}")

    response = query_anthropic(
        prompt=prompt, 
        system_prompt=system_prompt, 
        api_key=api_key, 
        verbose=verbose, 
        max_tokens=4000
    )
    
    # Extract JSON from response
    try:
        # Find JSON object in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            print("Failed to find JSON in response")
            return {}
        json_response = response[start_idx:end_idx]
        return json.loads(json_response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response: {response}")
        return {}

def extract_first_occurrence_ids(deduplicated_clusters: Dict[str, str]) -> List[int]:
    """
    Extracts the first occurrence ID from a topic ID.
    """
    return [int(k.split("-")[1]) for k in deduplicated_clusters.values()]

def batched_process_refusal_topics(all_runs_refusal_topics: Dict[str, List[str]], num_topics_per_batch: int, debug: bool = False) -> Tuple[Dict[str, str], List[int]]:
    """
    Process refusal topics in batches to handle context window limitations.
    
    Args:
        all_runs_refusal_topics: Dictionary mapping run titles to their refusal topics
        num_topics_per_batch: Number of topics to process in each batch
        
    Returns:
        Dictionary mapping unique cluster names to their first occurrence topic ID
    """
    # Create batches of topics
    batches = create_batched_topics(all_runs_refusal_topics, num_topics_per_batch)
    print(f"Created {len(batches)} batches with {num_topics_per_batch} topics per batch")
    
    # Process each batch
    all_clusters = {}
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        system_prompt, content_prompt = get_batch_summary_prompt()
        batch_clusters = process_batch_with_anthropic(system_prompt, content_prompt, batch)
        all_clusters.update(batch_clusters)
        if debug:
            if i >= 1:
                break

    
    # Deduplicate clusters
    print(f"Deduplicating {len(all_clusters)} clusters")
    system_prompt, content_prompt = get_deduplication_prompt()
    deduplicated_clusters = process_batch_with_anthropic(system_prompt, content_prompt, all_clusters)
    first_occurrence_ids = extract_first_occurrence_ids(deduplicated_clusters)
    
    return deduplicated_clusters, first_occurrence_ids


def save_results(results: Dict[str, str], run_titles: str):
    """
    Saves the clustered results to a file.
    
    Args:
        results: Dictionary mapping unique cluster names to their first occurrence topic ID
        run_titles: String representing the concatenated run titles
    """
    with open(os.path.join(RESULT_DIR, f"summarize_crawl_{run_titles}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save a readable text version
    with open(os.path.join(RESULT_DIR, f"summarize_crawl_{run_titles}.txt"), "w") as f:
        f.write("CLUSTERED REFUSAL TOPICS:\n\n")
        for cluster_name, topic_id in results.items():
            f.write(f"{cluster_name}: {topic_id}\n")

def plot_first_occurrence_ids(first_occurrence_ids: List[int], run_titles: str):
    """
    Plots the first occurrence IDs.
    """
    plt.step(range(len(first_occurrence_ids)), first_occurrence_ids, where='pre')
    plt.xlabel('Topic Number')
    plt.ylabel('Number of unique clusters')
    plt.title('First Occurrence IDs of Refusal Topics')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, f"first_occurrence_ids_{run_titles}.png"))


def generate_summary_table(title_mapping: Dict[str, str], num_topics_per_batch: int = 250, debug: bool = False):
    """
    Main function to run the entire process.
    """
    # Collect refusal topics from all runs
    all_runs_refusal_topics, all_run_titles, all_run_paths = collect_all_refusal_topics(title_mapping)
    
    # Create a unique name for the output file
    run_titles = "--".join(all_run_titles)
    all_run_path_names = [n.replace(".json", "") for n in all_run_paths]
    run_titles += "---" + "--".join(all_run_path_names)
    
    # Process the refusal topics in batches
    results, first_occurrence_ids = batched_process_refusal_topics(all_runs_refusal_topics, num_topics_per_batch, debug)
    plot_first_occurrence_ids(first_occurrence_ids, run_titles)
    # Save the results
    save_results(results, run_titles)
    print(f"Processing complete. Results saved to summarize_crawl_{run_titles}.json")
    print(f"First occurrence IDs saved to first_occurrence_ids_{run_titles}.png")


if __name__ == "__main__":
    title_mapping = {
        "ttp": "crawler_log_20250321_225822_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q8.json",
        # "deepseek-70B-init-q4": "crawler_log_20250224_041145_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
        # "deepseek-70B-init-q8": "crawler_log_20250224_041716_DeepSeek-R1-Distill-Llama-70B_1samples_100000crawls_Truefilter.json",
        "tus": "crawler_log_20250323_180420_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q0.json",
        "tua": "crawler_log_20250323_180018_Llama-3.1-Tulu-3-8B-SFT_1samples_100000crawls_Truefilter_q0.json",
    }
    generate_summary_table(title_mapping)

