import re
import os
import random
from tqdm import trange, tqdm
import json
from typing import List, Dict, Tuple
from copy import copy

from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
from spacy.language import Language
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from core.generation_utils import batch_generate, compute_embeddings, batch_compute_openai_embeddings
from core.topic_queue import TopicQueue, Topic
from core.crawler_config import CrawlerConfig

EPS = 1e-10
nvmlInit()


class CrawlerStats:
    def __init__(self):
        # Cumulative counters
        self.total_all = 0  # All topics generated
        self.total_deduped = 0  # These are all HEAD topics (topics after deduplication)
        self.total_refusals = 0  # All deduplicated topics that yield refusals

        # History tracking
        self.all_per_step = []
        self.deduped_per_step = []
        self.refusal_per_step = []
        self.cluster_sizes_per_step = []

    def log_step(
        self,
        new_topics_all: int,
        new_topics_deduped: int,
        new_topics_refusals: float,
        current_clusters: List[List[Topic]],
    ):
        """Log statistics for current step"""
        self.total_all += new_topics_all
        self.total_deduped += new_topics_deduped
        self.total_refusals += new_topics_refusals
        self.all_per_step.append(new_topics_all)
        self.deduped_per_step.append(new_topics_deduped)
        self.refusal_per_step.append(new_topics_refusals)
        self.cluster_sizes_per_step.append([len(cluster) for cluster in current_clusters])

    def get_current_metrics(self) -> dict:
        """Get current state of metrics"""
        return {
            "total_all": self.total_all,
            "total_deduped": self.total_deduped,
            "total_refusals": sum(self.refusal_per_step),
            "avg_refusal_rate": (sum(self.refusal_per_step) / (self.total_all + EPS)),
            "current_step": len(self.all_per_step),
            "largest_cluster": (
                max(self.cluster_sizes_per_step[-1]) if self.cluster_sizes_per_step else 0
            ),
        }

    def visualize_cumulative_topic_count(
        self, save_path: str = None, show_all_topics: bool = False, title: str = "Cumulative topic and refusal count"
    ):
        cumulative_generations = torch.cumsum(torch.tensor(self.all_per_step), dim=0)
        cumulative_topics = torch.cumsum(torch.tensor(self.deduped_per_step), dim=0)
        cumulative_refusals = torch.cumsum(torch.tensor(self.refusal_per_step), dim=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.grid(zorder=-1)
        ax.scatter(cumulative_generations, cumulative_topics, label="Unique topics", zorder=10)
        ax.scatter(
            cumulative_generations, cumulative_refusals, label="Refused unique topics", zorder=10
        )
        ax.set_xlabel("Total crawled topics")
        ax.set_ylabel("Total crawled topics after filter")
        ax.set_title(title)
        ax.legend()
        if save_path is not None:
            plt.savefig(save_path)
        return fig

    def to_dict(self):
        """Convert the crawler stats to a dictionary representation."""
        stats_dict = {
            "cumulative": {
                "total_all": self.total_all,
                "total_deduped": self.total_deduped,
                "total_refusals": self.total_refusals,
            },
            "current_metrics": self.get_current_metrics(),
            "history": {
                "all_per_step": self.all_per_step,
                "deduped_per_step": self.deduped_per_step,
                "refusal_per_step": self.refusal_per_step,
                "cluster_sizes_per_step": self.cluster_sizes_per_step,
            },
        }
        return stats_dict

    def save(self, filename: str):
        """Save the crawler stats to a JSON file."""
        stats_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(stats_dict, f)

        return stats_dict

    @classmethod
    def load(cls, stats_dict: dict):
        """Load the crawler stats from a dictionary."""
        crawler_stats = cls()
        crawler_stats.total_all = stats_dict["cumulative"]["total_all"]
        crawler_stats.total_deduped = stats_dict["cumulative"]["total_deduped"]
        crawler_stats.total_refusals = stats_dict["cumulative"]["total_refusals"]
        crawler_stats.all_per_step = stats_dict["history"]["all_per_step"]
        crawler_stats.deduped_per_step = stats_dict["history"]["deduped_per_step"]
        crawler_stats.refusal_per_step = stats_dict["history"]["refusal_per_step"]
        crawler_stats.cluster_sizes_per_step = stats_dict["history"]["cluster_sizes_per_step"]
        return crawler_stats

    def __repr__(self):
        return f"CrawlerStats(total_all={self.total_all}, total_deduped={self.total_deduped}, all_per_step={self.all_per_step}, deduped_per_step={self.deduped_per_step}, refusal_per_step={self.refusal_per_step}, cluster_sizes_per_step={self.cluster_sizes_per_step})"


class Crawler:
    def __init__(self, crawler_config: CrawlerConfig, save_filename: str) -> None:

        self.config = crawler_config
        self.queue = TopicQueue()
        self.head_embedding_CD: Tensor = None  # will be initialized in initialize_head_embeddings
        # NOTE Model, Tokenizer, Device are not saved to the Object to reduce overhead

        self.stats = CrawlerStats()

        self.numbered_list_pattern = re.compile(r"\d+\.\s*(.*?)(?:\n|$)")
        self.chinese_pattern = re.compile(r"[\u4e00-\u9fff]")

        self.save_filename = save_filename
        self.save(save_filename)  # Already testing at initialization whether saving works

    def _extract_from_numbered_list(self, text: str) -> List[str]:
        """Extract topics from a text that contains a numbered list."""
        extracted_list = self.numbered_list_pattern.findall(text)
        extracted_list = list(dict.fromkeys(extracted_list))  # Remove exact duplicates
        extracted_list = extracted_list[: self.config.max_extracted_topics_per_generation]
        return extracted_list

    def _has_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters."""
        return bool(self.chinese_pattern.search(text))

    def _translate_chinese_english(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        topics: List[Topic],
    ) -> List[Topic]:
        """Given a list of texts, translate the texts with chinese characters to english. Do not translate others.
        Changes the order of topics in the batch to [english] + [chinese]"""
        # check for chinese characters
        chinese_topics, topic_indices = [], []
        for i, topic in enumerate(topics):
            topic.is_chinese = self._has_chinese(topic.raw)
            if topic.is_chinese:
                chinese_topics.append(topic)
                topic_indices.append(i)

        # translate the subset with chinese characters in a single batch
        for batch_start in range(0, len(chinese_topics), self.config.generation_batch_size):
            batch_end = batch_start + self.config.generation_batch_size
            chinese_topic_B = chinese_topics[batch_start:batch_end]
            topic_indices_B = topic_indices[batch_start:batch_end]
            chinese_raw_B = [t.raw for t in chinese_topic_B]
            zh_en_ids_B = tokenizer_zh_en(
                chinese_raw_B, padding=True, truncation=True, return_tensors="pt"
            ).to(model_zh_en.device)
            with torch.inference_mode():
                translated_ids_B = model_zh_en.generate(**zh_en_ids_B)
            translated_str_B = tokenizer_zh_en.batch_decode(
                translated_ids_B, skip_special_tokens=True
            )
            for translation, idx in zip(translated_str_B, topic_indices_B):
                topics[idx].translation = translation
                topics[idx].text = translation  # Override text with translation

        return topics

    def _semantic_filter(self, model_spacy_en: Language, topics: List[Topic]) -> List[Topic]:
        # Process in batches using the configured batch size
        for batch_start in range(0, len(topics), self.config.generation_batch_size):
            batch_end = batch_start + self.config.generation_batch_size
            batch_topics = topics[batch_start:batch_end]

            # Process batch of texts together
            docs = model_spacy_en.pipe([topic.text for topic in batch_topics])

            # Update each topic's text with filtered tokens
            for topic, doc in zip(batch_topics, docs):
                meaningful_tokens = [
                    token.text
                    for token in doc
                    if token.tag_[:2] in set(self.config.allowed_spacy_tags)
                ]
                topic.text = " ".join(meaningful_tokens)

        return topics

    def _regex_filter(self, topics: List[Topic]) -> List[Topic]:
        for topic in topics:
            item = topic.text
            item = item.lower()
            item = item.strip(" ./:\",'()[]")
            item = item.replace(".", "")  # remove dots
            item = " ".join(
                word for word in item.split() if len(word) > 1
            )  # remove single characters
            topic.text = item
        return topics

    def _remove_words(self, topics: List[Topic]) -> List[Topic]:
        """
        Remove filtered words and duplicates from a list of texts.

        Args:
            texts: List of strings to process

        Returns:
            List of processed strings with filtered words removed
        """
        if not topics:
            return []

        for topic in topics:
            if not topic.text:
                continue

            # duplicate removal
            words = list(
                dict.fromkeys(
                    word
                    for word in topic.text.split()
                    if word not in set(self.config.regex_filter_global)
                )
            )

            # Remove filtered words from start and end
            while words and words[0] in set(self.config.regex_filter_start_end_only):
                words.pop(0)
            while words and words[-1] in set(self.config.regex_filter_start_end_only):
                words.pop()

            topic.text = " ".join(words)
        return topics

    def _split_at_comma(self, topics: List[Topic]) -> List[Topic]:
        for topic in topics:
            if "," in topic.text:
                splitted_text = topic.text.split(",")
                topic.text = splitted_text[0]
                for item in splitted_text[1:]:
                    topics.append(
                        Topic(text=item, raw=topic.raw, translation=topic.translation)
                    )
        return topics

    def extract_and_format(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        model_spacy_en: Language,
        generations: List[str],
        parent_ids: List[int],
        verbose: bool = False,
    ) -> List[Topic]:
        formatted_topics = []
        for gen, pid in zip(generations, parent_ids):
            extracted_items = self._extract_from_numbered_list(gen)
            for item in extracted_items:
                formatted_topics.append(Topic(text=item, raw=item, translation=None, parent_id=pid))
            
        if len(formatted_topics) == 0:
            print(f"Warning. No topics found in this generation:\n{generations}\n\n")
            return []

        self._translate_chinese_english(model_zh_en, tokenizer_zh_en, formatted_topics)
        self._regex_filter(formatted_topics)
        self._semantic_filter(model_spacy_en, formatted_topics)
        self._remove_words(formatted_topics)
        self._split_at_comma(formatted_topics)

        if verbose:
            print(f"formatted topics:\n{formatted_topics}\n\n")
        return formatted_topics

    def deduplicate_by_embedding_cossim(
        self,
        tokenizer_emb: AutoTokenizer,
        model_emb: AutoModel,
        formatted_topics: List[Topic],
        verbose: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Finds novel head topics in incoming batch and updates the self.queue.head_embedding. New topics are marked as heads."""
        if (
            self.head_embedding_CD is None
        ):  # Initializing it here so we don't have to pass in model_emb and device to the __init__ method
            self.initialize_head_embeddings(model_emb, tokenizer_emb, use_openai_embeddings=False)
        if formatted_topics == []:
            return formatted_topics

        formatted_text = [t.text for t in formatted_topics]

        with torch.inference_mode(), torch.no_grad():
            batch_embeddings_BD = compute_embeddings(tokenizer_emb, model_emb, formatted_text)

            # Update topic
            for topic, embedding_D in zip(formatted_topics, batch_embeddings_BD):
                cossim_C = embedding_D @ self.head_embedding_CD.T
                max_cossim, cluster_idx = torch.max(cossim_C, dim=-1)
                is_head = max_cossim < self.config.cossim_thresh
                if is_head:
                    self.head_embedding_CD = torch.cat(
                        (self.head_embedding_CD, embedding_D[None, :]), dim=0
                    )

                topic.cluster_idx = cluster_idx.item()
                topic.cossim_to_head = max_cossim.item()
                topic.is_head = is_head.item()

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics:\n")
            for t in new_head_topics:
                print(f"{t.text}\n{t.raw}\n{t.translation}\n\n")
        return formatted_topics

    def deduplicate_oai(
        self,
        openai_client,
        openai_emb_model: str,
        formatted_topics: List[Topic],
        verbose: bool = False,
    ) -> List[Topic]:
        """
        Finds novel head topics in incoming batch using OpenAI embeddings and updates the self.queue.head_embedding.
        New topics are marked as heads.
        
        Args:
            openai_client: OpenAI client
            openai_emb_model: Name of the OpenAI embedding model to use
            formatted_topics: List of topics to deduplicate
            verbose: Whether to print verbose output
            
        Returns:
            List[Topic]: The input topics with is_head, cluster_idx, and cossim_to_head fields updated
        """
        if self.head_embedding_CD is None:
            self.initialize_head_embeddings(openai_client, openai_emb_model, use_openai_embeddings=True)
        
        if formatted_topics == []:
            return formatted_topics

        formatted_text = [t.text for t in formatted_topics]

        # Compute embeddings using OpenAI API
        batch_embeddings_BD = batch_compute_openai_embeddings(openai_client, openai_emb_model, formatted_text)

        # Update topic
        for topic, embedding_D in zip(formatted_topics, batch_embeddings_BD):
            cossim_C = embedding_D @ self.head_embedding_CD.T
            max_cossim, cluster_idx = torch.max(cossim_C, dim=-1)
            is_head = max_cossim < self.config.cossim_thresh
            if is_head:
                self.head_embedding_CD = torch.cat(
                    (self.head_embedding_CD, embedding_D[None, :]), dim=0
                )

            topic.cluster_idx = cluster_idx.item()
            topic.cossim_to_head = max_cossim.item()
            topic.is_head = is_head.item()

        if verbose:
            new_head_topics = [t for t in formatted_topics if t.is_head]
            print(f"new head topics (OpenAI):\n")
            for t in new_head_topics:
                print(f"{t.text}\n{t.raw}\n{t.translation}\n\n")
        
        return formatted_topics

    def is_refusal(self, text: str) -> bool:
        return any(refusal.lower() in text.lower() for refusal in set(self.config.refusal_messages))

    def check_refusal(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        selected_topics: List[Topic],
        user_message_templates: List[List[str]],
        force_thought_skip: bool = False,
        verbose: bool = False,
    ) -> List[Topic]:
        """Filtering incoming head topics for refusals."""

        if selected_topics == []:
            return selected_topics

        head_topics_raw = [t.raw for t in selected_topics if t.is_head]
        if len(head_topics_raw) == 0:
            return selected_topics

        # Preparation
        refusals = torch.zeros(len(head_topics_raw))
        responses = [[]]*len(head_topics_raw)
        flattened_templates = []
        for language, templates in user_message_templates.items():
            flattened_templates.extend(templates)

        # Accumulate refusal rates across templates
        for template in flattened_templates:
            generated_texts = batch_generate(
                model,
                tokenizer,
                head_topics_raw,
                template,
                max_new_tokens=self.config.refusal_max_new_tokens,
                force_thought_skip=force_thought_skip,
                verbose=verbose,
            )
            is_refusal_B = []
            for head_topic_idx, gen in enumerate(generated_texts):
                responses[head_topic_idx].append(gen)
                is_refusal = self.is_refusal(gen)
                is_refusal_B.append(is_refusal)
                if verbose:
                    print(f"\n\n")
                    print(f"refusal: {bool(is_refusal)}")
                    print(gen)

            is_refusal_B = torch.tensor(is_refusal, dtype=float)
            refusals += is_refusal_B
                    
        majority_refusal_count = 0.5 * len(user_message_templates)
        refusals = iter(refusals)
        responses = iter(responses)
        for topic in selected_topics:
            if topic.is_head:
                topic.is_refusal = bool(next(refusals) >= majority_refusal_count)
                topic.responses = next(responses)
        return selected_topics

    def batch_check_refusal(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        topics: List[Topic],
        verbose: bool = False,
    ) -> List[Topic]:
        """Check if the topics are refusals."""
        for batch_start in trange(
            0, len(topics), self.config.generation_batch_size, desc="Checking refusals"
        ):
            batch_topics = topics[batch_start : batch_start + self.config.generation_batch_size]
            topics = self.check_refusal(
                model,
                tokenizer,
                batch_topics,
                self.config.user_message_templates,
                self.config.do_force_thought_skip,
                verbose,
            )
        return topics

    def initialize_topics(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        initial_topics: List[str],
        verbose: bool = False,
    ) -> List[Topic]:
        """Initialize all initial topics as heads."""
        topics = []
        for i, topic_str in enumerate(initial_topics):
            topics.append(
                Topic(
                    text=topic_str,
                    raw=topic_str,
                    is_head=True,
                    cluster_idx=i,
                    cossim_to_head=1.0,
                    is_refusal=None,  # will be checked immediately below
                    parent_id=-5,
                )
            )
        topics = self.check_refusal(
            model=model,
            tokenizer=tokenizer,
            selected_topics=topics,
            user_message_templates=self.config.user_message_templates,
            force_thought_skip=self.config.do_force_thought_skip,
            verbose=verbose,
        )
        topics = self.queue.incoming_batch(topics)
        if verbose:
            for topic in topics:
                print(
                    f"new topic from initial topics:\n{topic.text}\n{topic.raw}\n{topic.translation}\nID:{topic.id}\nis_refusal:{topic.is_refusal}\n\n"
                )
        return topics

    def initialize_head_embeddings(self, model_emb: AutoModel, tokenizer_emb: AutoTokenizer, use_openai_embeddings: bool = False):
        self.head_embedding_CD = torch.zeros(
            0, model_emb.config.hidden_size, device=model_emb.device
        )  # [num_head_topics, hidden_size]
        for batch_start in range(
            0, self.queue.num_head_topics, self.config.load_embedding_batch_size
        ):
            batch_end = min(
                batch_start + self.config.load_embedding_batch_size, self.queue.num_head_topics
            )
            batch_topics = self.queue.head_topics[batch_start:batch_end]
            if use_openai_embeddings:
                batch_embeddings = batch_compute_openai_embeddings(
                    openai_client=self.config.openai_client,
                    openai_emb_model=self.config.openai_emb_model,
                    words=[t.text for t in batch_topics]
                )
            else:
                batch_embeddings = compute_embeddings(
                    tokenizer_emb, model_emb, [t.text for t in batch_topics]
                )
            self.head_embedding_CD = torch.cat((self.head_embedding_CD, batch_embeddings), dim=0)

    def crawl(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        filter_models: Dict,
        verbose: bool = False,
    ) -> List[str]:
        """Crawl the topics."""

        # TODO if meta in model name, force thought skip has to be false.

        if self.config.initial_topics:
            self.initialize_topics(
                model=model,
                tokenizer=tokenizer,
                initial_topics=self.config.initial_topics,
                verbose=verbose,
            )

        if self.config.do_filter_refusals:
            topic_seed_candidates = self.queue.head_refusal_topics
        else:
            topic_seed_candidates = self.queue.head_topics

        for crawl_step_idx in trange(self.config.num_crawl_steps, desc="Crawling topics"):
            # Get batch of topics
            if len(topic_seed_candidates) <= self.config.generation_batch_size:
                # Early stage filling the queue
                batch_topics = topic_seed_candidates
            else:
                # Randomly sample for topic diversity
                batch_topics = random.sample(
                    topic_seed_candidates, self.config.generation_batch_size
                )
            batch_topics_raw = [t.raw for t in batch_topics]
            parent_ids = [t.id for t in batch_topics] * self.config.num_samples_per_topic

            # Get user message templates
            # Starting condition is necessary, model output is not coherent for unfilled templates
            if batch_topics_raw == []:
                batch_topics_raw = [""]
                parent_ids = [-5]
                user_message_templates = self.config.fallback_user_message_templates
            else:
                # Draw a random english-chinese template pair
                user_message_templates = self.config.user_message_templates

            # Generate with prefilling
            for lang in user_message_templates.keys():
                for template in user_message_templates[lang]:
                    for thinking_message in self.config.crawler_thinking_messages[lang]:
                        torch.cuda.empty_cache()

                        print(f"\n## generating...")
                        # Choose the correct prefill based on the model
                        if "deepseek" in model.config._name_or_path:
                            prefills = {"thinking_message": thinking_message}
                        elif "meta" in model.config._name_or_path:
                            prefills = {"assistant_prefill": thinking_message}
                        else:
                            raise ValueError(f"Unsupported model: {model.config._name_or_path}. Only DeepSeek and Meta models are supported.")

                        generated_texts = batch_generate(
                            model,
                            tokenizer,
                            batch_topics_raw,
                            user_message_template=template,
                            force_thought_skip=False,
                            max_new_tokens=self.config.max_generated_tokens,
                            temperature=self.config.temperature,
                            tokenization_template=self.config.tokenization_template,
                            num_samples_per_topic=self.config.num_samples_per_topic,
                            verbose=verbose,
                            **prefills,
                        )

                        print(f"\n## formatting...")
                        new_topics = self.extract_and_format(
                            model_zh_en=filter_models["model_zh_en"],
                            tokenizer_zh_en=filter_models["tokenizer_zh_en"],
                            model_spacy_en=filter_models["model_spacy_en"],
                            generations=generated_texts,
                            parent_ids=parent_ids,
                            verbose=verbose,
                        )

                        print(f"\n## deduplicating...")
                        # Use OpenAI embeddings if available, otherwise fall back to local embeddings
                        if filter_models.get("has_openai", False):
                            new_topics = self.deduplicate_oai(
                                openai_client=filter_models["openai_client"],
                                openai_emb_model=filter_models["openai_emb_model"],
                                formatted_topics=new_topics,
                                verbose=verbose,
                            )
                        else:
                            new_topics = self.deduplicate_by_embedding_cossim(
                                tokenizer_emb=filter_models["tokenizer_emb"],
                                model_emb=filter_models["model_emb"],
                                formatted_topics=new_topics,
                                verbose=verbose,
                            )

                        if self.config.do_filter_refusals:
                            print(f"\n## filtering for refusal...")
                            new_topics = self.check_refusal(
                                model=model,
                                tokenizer=tokenizer,
                                selected_topics=new_topics,
                                user_message_templates=self.config.user_message_templates,
                                force_thought_skip=self.config.do_force_thought_skip,
                                verbose=verbose,
                            )

                        # Update queue
                        self.queue.incoming_batch(new_topics)
                        if verbose:
                            for topic in new_topics:
                                print(
                                    f"new topic from crawl step {crawl_step_idx}:\n{topic.text}\n{topic.raw}\n{topic.translation}\n\n"
                                )
                                # Record memory usage

                        gpu_handle = nvmlDeviceGetHandleByIndex(0)
                        gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
                        print(f"GPU Memory: {(gpu_info.used / 1024**2):.1f} MB")
                        print(f"RAM Usage: {(psutil.Process().memory_info().rss / 1024**2):.1f} MB")

                        # Log stats
                        self.stats.log_step(
                            new_topics_all=len(new_topics),
                            new_topics_deduped=sum(1 for t in new_topics if t.is_head),
                            new_topics_refusals=sum(1 for t in new_topics if t.is_refusal),
                            current_clusters=self.queue.cluster_topics,
                        )

            if crawl_step_idx % 2 == 0:
                self.save(self.save_filename)
            if self.queue.num_head_topics > self.config.max_crawl_topics:
                print(f"Topic queue has {len(self.queue.head_topics)} topics")
                break

        self.save(self.save_filename)
        return self.queue

    def to_dict(self):
        crawler_dict = {
            "stats": self.stats.to_dict(),
            "config": self.config.to_dict(),
            "queue": self.queue.to_dict(),
        }
        return crawler_dict

    def save(self, filename: str):
        crawler_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(crawler_dict, f)
        return crawler_dict

    @classmethod
    def load(cls, load_from_filename: str, save_to_filename: str):
        with open(load_from_filename, "r") as f:
            crawler_dict = json.load(f)

        crawler_config = CrawlerConfig(**crawler_dict["config"])
        crawler = cls(crawler_config, save_to_filename)
        crawler.queue = TopicQueue.load(crawler_dict["queue"])
        crawler.stats = CrawlerStats.load(crawler_dict["stats"])
        return crawler

    # NOTE: Needs update, potentially move to another experiment class
    # def _batch_generate_and_process(
    #     self,
    #     topics: List[str],
    #     experiment_name: str = "base",
    #     thinking_message: str = "",
    #     user_suffix: str = None,
    #     assistant_prefill: str = None,
    # ) -> Dict:
    #     """Common batch generation logic used across different methods"""
    #     for batch_start in tqdm(
    #         range(0, len(topics), self.config.generation_batch_size),
    #         desc=f"Generating responses with {experiment_name}",
    #     ):
    #         batch_topics = topics[batch_start : batch_start + self.config.generation_batch_size]
    #         user_messages = self.get_user_messages(self.config.refusal_user_message, batch_topics)

    #         input_ids = custom_batch_encoding(
    #             model_name=self.model_name,
    #             tokenizer=self.tokenizer,
    #             user_messages=user_messages,
    #             thinking_message=thinking_message,
    #             user_suffix=user_suffix,
    #             assistant_prefill=assistant_prefill,
    #             template=self.config.tokenization_template,
    #         )

    #         texts = batch_generate_from_tokens(
    #             model=self.model,
    #             tokenizer=self.tokenizer,
    #             input_ids_BL=input_ids,
    #             max_generation_length=self.config.max_generated_tokens,
    #             temperature=None,  # Greedy decoding for generation and experiments
    #         )

    #         for topic, text in zip(batch_topics, texts):
    #             text_key = f"generated_text_{experiment_name}"
    #             refusal_key = f"refusal_{experiment_name}"

    #             self.crawler_dict["exp"][topic].update(
    #                 {
    #                     text_key: text,
    #                     refusal_key: self.is_refusal(text),
    #                 }
    #             )

    #     return self.crawler_dict

    # def generate_responses(self, topics: List[str], thinking_message: str = "") -> Dict:
    #     return self._batch_generate_and_process(
    #         topics, experiment_name="base", thinking_message=thinking_message
    #     )

    # def generate_responses_base_model(self, topics: List[str], thinking_message: str = "") -> Dict:
    #     return self._batch_generate_and_process(
    #         topics, experiment_name="base_model", thinking_message=thinking_message
    #     )

    # def run_experiment(
    #     self,
    #     experiment_name: str,
    #     topics: List[str] = None,
    #     thinking_message: str = "",
    #     user_suffix: str = "",
    #     assistant_prefill: str = "",
    #     only_base_refusals: bool = True,
    # ) -> Dict:
    #     """Run an experiment with the given parameters.

    #     Args:
    #         experiment_name: Name of the experiment (e.g. "base", "ttf", "user_suffix")
    #         topics: List of topics to run experiment on. If None, uses all topics
    #         thinking_message: Optional thinking message to prepend
    #         user_suffix: Optional suffix to add to user message
    #         assistant_prefill: Optional prefill for assistant message
    #         only_base_refusals: If True, only run on topics that had refusal_base=True
    #     """
    #     if topics is None:
    #         if only_base_refusals:
    #             topics = [
    #                 topic
    #                 for topic, data in self.crawler_dict["exp"].items()
    #                 if data["refusal_base"]
    #             ]
    #         else:
    #             topics = list(self.crawler_dict["exp"].keys())

    #     # Mark non-refusal topics if running on base refusals
    #     if only_base_refusals:
    #         for topic, data in self.crawler_dict["exp"].items():
    #             if not data["refusal_base"]:
    #                 self.crawler_dict["exp"][topic].update({f"refusal_{experiment_name}": False})

    #     return self._batch_generate_and_process(
    #         topics,
    #         experiment_name=experiment_name,
    #         thinking_message=thinking_message,
    #         user_suffix=user_suffix,
    #         assistant_prefill=assistant_prefill,
    #     )

    # def get_stats(self) -> Dict:
    #     # Initialize base refusal count
    #     refusal_counts = {
    #         "base": sum(data["refusal_base"] for data in self.crawler_dict["exp"].values())
    #     }

    #     # Get list of all experiment types from a sample data point
    #     sample_data = next(iter(self.crawler_dict["exp"].values()))
    #     experiment_types = [
    #         key.split("refusal_")[-1]  # Get the last part after "refusal_"
    #         for key in sample_data.keys()
    #         if key.startswith("refusal_")
    #     ]

    #     # Initialize counters for each experiment
    #     for exp_type in experiment_types:
    #         refusal_counts[exp_type] = 0

    #     # Count refusals for each experiment (only for base refusal cases)
    #     for data in self.crawler_dict["exp"].values():
    #         if data["refusal_base"]:
    #             for exp_type in experiment_types:
    #                 refusal_counts[exp_type] += data[f"refusal_{exp_type}"]

    #     # Calculate conditional probabilities
    #     for exp_type in experiment_types:
    #         if exp_type != "base":  # Skip base to avoid base_given_base
    #             refusal_counts[f"{exp_type}_given_base"] = refusal_counts[exp_type] / (
    #                 refusal_counts["base"] + 1e-10
    #             )

    #     self.crawler_dict["stats"] = refusal_counts
    #     return refusal_counts

    # def save(self, filename: str):
    #     topic_dict = self.topic_queue.export_to_dict()
    #     topic_dict["stats"] = self.stats
    #     topic_dict["config"] = self.config
    #     topic_dict["model_name"] = self.model_name
    #     topic_dict["device"] = self.device
    #     with open(os.path.join(self.config.output_dir, filename), "w", encoding="utf-8") as f:
    #         json.dump(topic_dict, f)


def get_run_name(model_path: str, crawler_config: CrawlerConfig):
    model_name = model_path.split("/")[-1]
    run_name = (
        "crawler_log"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{model_name}"
        f"_{crawler_config.num_samples_per_topic}samples"
        f"_{crawler_config.num_crawl_steps}crawls"
        f"_{crawler_config.do_filter_refusals}filter"
    )
    return run_name
