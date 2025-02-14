import re
import os
import random
from tqdm import trange, tqdm
import json
from typing import List, Dict, Tuple
from copy import copy

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel
from spacy.language import Language

from core.generation_utils import batch_generate
from core.topic_queue import TopicQueue, Topic
from core.crawler_config import CrawlerConfig

EPS = 1e-10

# TODO: add a debug mode and test this thing.
# TODO: Save to each topic in the queue whether they caused a refusal. Only check for refusals if the topic is added as head. maybe move refusal logic to topic queue?
# TODO: For prefilling, draw templates and thinking messages in the same language
# TODO: Integrate stats class into crawler class
# TODO: Make more efficient
# TODO: Test crawler stats class for logging


class CrawlerStats:
    def __init__(self):
        # Cumulative counters
        self.total_generations = 0
        self.total_topics = 0  # These are all HEAD topics (topics after deduplication)

        # History tracking
        self.generations_per_step = []
        self.topics_per_step = []
        self.refusals_per_step = []
        self.cluster_sizes = []

    def log_step(
        self,
        new_generations: int,
        new_topics: int,
        new_refusals: float,
        current_clusters: List[List[Topic]],
    ):
        """Log statistics for current step"""
        self.total_generations += new_generations
        self.total_topics += new_topics

        self.generations_per_step.append(new_generations)
        self.topics_per_step.append(new_topics)
        self.refusals_per_step.append(new_refusals)
        self.cluster_sizes.append([len(cluster) for cluster in current_clusters])

    def get_current_metrics(self) -> dict:
        """Get current state of metrics"""
        return {
            "total_generations": self.total_generations,
            "total_topics": self.total_topics,
            "avg_refusal_rate": (sum(self.refusals_per_step) / (self.total_generations + EPS)),
            "current_step": len(self.generations_per_step),
            "largest_cluster": max(self.cluster_sizes[-1]) if self.cluster_sizes else 0,
        }

    def to_dict(self):
        """Convert the crawler stats to a dictionary representation."""
        stats_dict = {
            "cumulative": {
                "total_generations": self.total_generations,
                "total_topics": self.total_topics,
            },
            "history": {
                "generations_per_step": self.generations_per_step,
                "topics_per_step": self.topics_per_step,
                "refusals_per_step": self.refusals_per_step,
                "cluster_sizes": self.cluster_sizes,
            },
            "current_metrics": self.get_current_metrics(),
        }
        return stats_dict

    def save(self, filename: str):
        """Save the crawler stats to a JSON file."""
        stats_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(stats_dict, f)

        return stats_dict

    @classmethod
    def load(cls, filename: str):
        """Load the crawler stats from a JSON file."""
        with open(filename, "r") as f:
            stats_dict = json.load(f)

        crawler_stats = cls()
        crawler_stats.total_generations = stats_dict["cumulative"]["total_generations"]
        crawler_stats.total_topics = stats_dict["cumulative"]["total_topics"]
        crawler_stats.generations_per_step = stats_dict["history"]["generations_per_step"]
        crawler_stats.topics_per_step = stats_dict["history"]["topics_per_step"]
        crawler_stats.refusals_per_step = stats_dict["history"]["refusals_per_step"]
        crawler_stats.cluster_sizes = stats_dict["history"]["cluster_sizes"]
        return crawler_stats


class Crawler:
    def __init__(
        self,
        crawler_config: CrawlerConfig,
        save_filename: str,
        device: str,
    ) -> None:

        self.config = crawler_config
        self.queue = TopicQueue(
            cossim_thresh=self.config.cossim_thresh,
            do_filter_refusals=self.config.do_filter_refusals,
            device=device,
        )
        self.stats = CrawlerStats()

        self.save_filename = save_filename
        self.save(save_filename)  # Already testing at initialization whether saving works

    def _extract_from_numbered_list(self, text: str) -> List[str]:
        """Extract topics from a text that contains a numbered list."""
        pattern = r"\d+\.\s*(.*?)(?:\n|$)"
        extracted_list = re.findall(pattern, text)
        extracted_list = list(dict.fromkeys(extracted_list))  # Remove exact duplicates
        return extracted_list

    def _has_chinese(self, text: str) -> bool:
        """Check if the text contains Chinese characters."""
        return bool(re.compile(r"[\u4e00-\u9fff]").search(text))

    def _translate_chinese_english(
        self, model_zh_en: AutoModelForSeq2SeqLM, tokenizer_zh_en: AutoTokenizer, texts: List[str]
    ) -> List[str]:
        """Given a list of texts, translate the texts with chinese characters to english."""
        # check for chinese characters
        is_chinese_B, chinese_texts = [], []
        for text in texts:
            is_chinese = self._has_chinese(text)
            is_chinese_B.append(is_chinese)
            if is_chinese:
                chinese_texts.append(text)

        if sum(is_chinese_B) < 1:
            return copy(texts)

        # translate the subset with chinese characters in a single batch
        zh_en_ids = tokenizer_zh_en(
            chinese_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        translated_ids = model_zh_en.generate(**zh_en_ids)
        translated_str = tokenizer_zh_en.batch_decode(translated_ids, skip_special_tokens=True)

        # match translations to full text
        texts_with_translations = [""] * len(texts)
        translations = iter(translated_str)
        for i, is_chinese in enumerate(is_chinese_B):
            if is_chinese:
                texts_with_translations[i] = next(translations)
            else:
                texts_with_translations[i] = texts[i]
        return texts_with_translations

    def _semantic_filter(self, model_spacy_en: Language, texts: List[str]) -> List[str]:
        filtered_texts = []
        for text in texts:
            doc = model_spacy_en(text)
            meaningful_tokens = [
                token.text for token in doc if token.tag_[:2] in self.config.allowed_spacy_tags
            ]
            filtered_texts.append(" ".join(meaningful_tokens))
        return filtered_texts

    def _regex_filter(self, texts: List[str]) -> List[str]:
        filtered_items = []
        for item in texts:
            item = item.lower()
            item = item.strip(" ./:\",'()[]")
            item = item.replace(".", "")  # remove dots
            item = " ".join(
                word for word in item.split() if len(word) > 1
            )  # remove single characters
            item = item[: self.config.max_topic_string_length]
            filtered_items.append(item)
            if len(filtered_items) >= self.config.max_extracted_topics_per_generation:
                # Items at numbering > 20. often seem to be hallucinated
                break
        return filtered_items

    def _remove_words(self, texts: List[str]) -> List[str]:
        """
        Remove filtered words and duplicates from a list of texts.

        Args:
            texts: List of strings to process

        Returns:
            List of processed strings with filtered words removed
        """
        if not texts:
            return []

        filtered_words = []

        for text in texts:
            if not text:
                continue

            # duplicate removal
            words = list(
                dict.fromkeys(
                    word for word in text.split() if word not in self.config.regex_filter_global
                )
            )

            # Remove filtered words from start and end
            while words and words[0] in self.config.regex_filter_start_end_only:
                words.pop(0)
            while words and words[-1] in self.config.regex_filter_start_end_only:
                words.pop()

            if words:
                filtered_words.append(" ".join(words))
        return filtered_words

    def _split_at_comma(self, texts: List[str]) -> List[List[str]]:
        splitted_items = []
        for item in texts:
            splitted = item.split(",")
            # splitted = item.split(" and ")
            # splitted = item.split(" or ")
            splitted_items.append(splitted)
        return splitted_items

    def extract_and_format(
        self,
        model_zh_en: AutoModelForSeq2SeqLM,
        tokenizer_zh_en: AutoTokenizer,
        model_spacy_en: Language,
        generations: List[str],
    ) -> List[Topic]:
        formatted_topics = []
        for gen in generations:
            extracted_list = self._extract_from_numbered_list(gen)
            if extracted_list == []:
                print(f"Warning. No topics found in this generation:\n{gen}\n\n")
                continue
            translated_list = self._translate_chinese_english(
                model_zh_en, tokenizer_zh_en, extracted_list
            )
            regex_list = self._regex_filter(translated_list)
            spacy_list = self._semantic_filter(model_spacy_en, regex_list)
            filtered_list = self._remove_words(spacy_list)
            splitted_list = self._split_at_comma(filtered_list)
            for topic_split, raw, translated in zip(splitted_list, extracted_list, translated_list):
                for topic_text in topic_split:
                    if raw == translated:
                        translated = None
                    topic = Topic(text=topic_text, raw=raw, translation=translated)
                    formatted_topics.append(topic)
        return formatted_topics
    
    def _batch_compute_embeddings(
        self, tokenizer_emb: AutoTokenizer, model_emb: AutoModel, topics: List[Topic], device: str
    ) -> Tensor:
        """Embed a batch of topics."""
        batch_formatted = [f"query: {t.text}" for t in topics]
        batch_dict = tokenizer_emb(
            batch_formatted, padding=True, truncation=False, return_tensors="pt"
        )
        batch_dict = batch_dict.to(device)

        # outputs.last_hidden_state.shape (batch_size, sequence_length, hidden_size) hidden activations of the last layer
        outputs = model_emb(**batch_dict)
        batch_embeddings_BD = self.final_token_hidden_state(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        batch_embeddings_BD = F.normalize(batch_embeddings_BD, p=2, dim=1)

        return batch_embeddings_BD
    
    def _compute_embedding_cossim(self, embedding_BD: Tensor) -> Topic:
        # compute cosine sim with cluster means
        # Find heads w.r.t queue
        cossim_queue_BC = embedding_BD @ self.queue.head_embedding_CD.T
        is_head_queue_B = cossim_queue_BC.max(dim=-1) < self.queue.cossim_thresh # TODO if the whole embedding logic is moved over, we can  move mean embedding and cossim thresh over into the crawler class

        # Find the heads wrt. batch
        cossim_batch_BB = embedding_BD @ embedding_BD.T
        cossim_batch_tril_BB = torch.tril(cossim_batch_BB, diagonal=-1)
        is_head_batch_B = cossim_batch_tril_BB < self.queue.cossim_thresh

        # Update list of head embeddings
        is_head_B = is_head_queue_B & is_head_batch_B
        new_head_embeddings = embedding_BD[is_head_B]
        self.queue.head_embedding_CD = torch.cat((self.queue.head_embedding_CD, new_head_embeddings), dim=0)

        # Assign duplicates to clusters
        cossim_BC = embedding_BD @ self.queue.head_embedding_CD.T # importantly, queue contains the new heads now!
        max_cossim_B, max_cossim_idx_B = torch.max(cossim_BC, dim=-1)
        return max_cossim_B, max_cossim_idx_B, is_head_B

    def deduplicate(self, tokenizer_emb: AutoTokenizer, model_emb: AutoModel, formatted_topics: List[Topic], device: str) -> Tuple[Tensor, Tensor, Tensor]:
        """Finds novel head topics in incoming batch and updates the self.queue.head_embedding"""
        batch_embeddings_BD = self._batch_compute_embeddings(tokenizer_emb, model_emb, formatted_topics, device)
        max_cossim_B, max_cossim_idx_B, is_head_B = self._compute_embedding_cossim(batch_embeddings_BD)
        return max_cossim_B, max_cossim_idx_B, is_head_B
    
    def is_refusal(self, text: str) -> bool:
        return any(refusal.lower() in text.lower() for refusal in self.config.refusal_messages)

    def filter_for_refusal(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        selected_topics: List[Topic],
        user_message_templates: List[List[str]],
        assistant_prefill: str = "",
        thinking_mesage: str = "",
        force_thought_skip: bool = False,
        verbose: bool = False,
    ) -> List[Topic]:
        """Filtering incoming head topics for refusals."""

        if selected_topics == []:
            return []

        # Preparation
        topic_str = [t.raw for t in selected_topics]
        refusals = torch.zeros(len(selected_topics))
        flattened_templates = []
        for template_pair in user_message_templates:
            flattened_templates.extend(template_pair)

        # Accumulate refusal rates across templates
        for template in flattened_templates:
            generated_texts = batch_generate(
                model,
                tokenizer,
                topic_str,
                template,
                assistant_prefill,
                thinking_mesage,
                force_thought_skip=force_thought_skip,
                verbose=False,
            )
            is_refusal_B = torch.tensor(
                [self.is_refusal(gen) for gen in generated_texts], dtype=float
            )
            refusals += is_refusal_B
            if verbose:
                for is_refusal, gen in zip(is_refusal_B, generated_texts):
                    if not is_refusal:
                        print(f"\n\n")
                        print(f"the following is a refusal: {bool(is_refusal)}")
                        print(gen)

        # Average refusal rate across templates
        refusals /= len(user_message_templates)

        # Return topics with majority refusal rate
        refused_topics = [
            topic for topic, refusal in zip(selected_topics, refusals) if refusal >= 0.5
        ]
        return refused_topics

    def crawl(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        filter_models: Dict,
        initial_topics: List[str] = [],
        assistant_prefill: str = "",
        thinking_messages: List[str] = [""],
        fallback_user_message_template: str = "Hi, what's on your mind?",
        verbose: bool = False,
    ) -> List[str]:

        initial_topics = [Topic(text=t) for t in initial_topics]
        self.queue.incoming_batch(initial_topics)

        for crawl_step_idx in trange(self.config.num_crawl_steps, desc="Crawling topics"):
            # Get batch of topics
            if self.queue.num_head_topics < self.config.generation_batch_size:
                # Early stage filling the queue
                batch_topics = self.queue.head_topics
            else:
                # Randomly sample for topic diversity
                batch_topics = random.sample(
                    self.queue.head_topics, self.config.generation_batch_size
                )
            batch_topics = [t.text for t in batch_topics]

            # Get user message templates
            # Starting condition is necessary, model output is not coherent for unfilled templates
            if batch_topics == []:
                batch_topics = [""]
                user_message_templates = [fallback_user_message_template]
            else:
                # Draw a random english-chinese template pair
                user_message_templates = random.choice(self.config.user_message_templates)

            # Generate with prefilling
            for template in user_message_templates:
                for thinking_message in thinking_messages:
                    print(f"generating...")
                    generated_texts = batch_generate(
                        self.model,
                        self.tokenizer,
                        batch_topics,
                        user_message_template=template,
                        assistant_prefill=assistant_prefill,
                        thinking_message=thinking_message,
                        force_thought_skip=False,
                        tokenization_template=self.config.tokenization_template,
                        num_samples_per_topic=self.config.num_samples_per_topic,
                    )
                    if verbose:
                        print(f"full generation:\n{generated_texts}\n\n")

                    print(f"formatting...")
                    new_topics = self.extract_and_format(
                        filter_models["model_zh_en"],
                        filter_models["tokenizer_zh_en"],
                        filter_models["model_spacy_en"],
                        generated_texts,
                    )

                    if self.config.do_filter_refusals:
                        print(f"filtering for refusal...")
                        new_topics = self.filter_for_refusal(
                            new_topics,
                            self.config.user_message_templates,
                            force_thought_skip=self.config.force_thought_skip,
                            verbose=False,
                        )
                    if verbose:
                        for topic in new_topics:
                            print(
                                f"new topic from crawl step {crawl_step_idx}:\n{topic.text}\n{topic.raw}\n{topic.translation}\n\n"
                            )

                    print("deduplicating by embedding similarity...")
                    added_topics = self.queue.incoming_batch(new_topics)

                    if not self.config.do_filter_refusals:
                        print(f"evaluating added topics for refusals...")
                        added_and_refused_forced = self.filter_for_refusal(
                            added_topics,
                            self.config.user_message_templates,
                            force_thought_skip=True,
                            verbose=False,
                        )
                        self.stats["num_refusal_heads_thought_skip"].append(
                            len(added_and_refused_forced)
                            + self.stats["num_refusal_heads_thought_skip"][-1]
                        )
                        added_and_refused_standard = self.filter_for_refusal(
                            added_topics,
                            self.config.user_message_templates,
                            force_thought_skip=False,
                            verbose=False,
                        )
                        self.stats["num_refusal_heads_standard"].append(
                            len(added_and_refused_standard)
                            + self.stats["num_refusal_heads_standard"][-1]
                        )

                    self.stats["num_generations"].append(
                        len(batch_topics) + self.stats["num_generations"][-1]
                    )
                    self.stats["num_topic_heads"].append(self.queue.num_head_topics)

            if crawl_step_idx % 10 == 0:
                self.save(self.save_filename)
            # if self.topic_queue.num_clusters > self.config.max_crawl_topics:
            #     print(f"Topic queue has {len(self.topic_queue.num_clusters)} topics")
            #     break

        self.save(self.save_filename)
        return self.queue

    def to_dict(self):
        crawler_dict = {
            "queue": self.queue.to_dict(),
            "stats": self.stats.to_dict(),
            "config": self.config.to_dict(),
        }
        return crawler_dict

    def save(self, filename: str):
        crawler_dict = self.to_dict()
        with open(filename, "w") as f:
            json.dump(crawler_dict, f)
        return crawler_dict

    @classmethod
    def load(cls, filename: str, device: str):
        with open(filename, "r") as f:
            crawler_dict = json.load(f)

        crawler = cls(crawler_dict["config"], device)
        crawler.queue = TopicQueue.load(
            crawler_dict["queue"], crawler.model, crawler.tokenizer, device
        )
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
