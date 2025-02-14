from dataclasses import dataclass
from typing import List
import json

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


@dataclass
class Topic:
    text: str
    raw: str = None
    translation: str = None
    is_head: bool = None
    is_refusal: bool = None
    cossim_to_head: float = None
    cluster_idx: int = None

    def to_dict(self):
        return {
            "text": self.text,
            "raw": self.raw,
            "translation": self.translation,
            "is_head": self.is_head,
            "cossim_to_head": self.cossim_to_head,
            "cluster_idx": self.cluster_idx,
        }


class TopicQueue:
    def __init__(self, cossim_thresh: float, do_filter_refusals: bool, device: str):
        # Parameters
        self.do_filter_refusals: bool = do_filter_refusals
        self.cossim_thresh: float = cossim_thresh

        # Track clusters
        self.head_topics: List[Topic] = []
        self.cluster_topics: List[List[Topic]] = []
        self.cluster_cossims: List[List[float]] = []
        self.non_refusal_topics: List[Topic] = []

        # Stats
        self.num_head_topics: int = 0
        self.num_topics_per_cluster: List[int] = []
        self.num_non_refusal_topics: int = 0
        self.num_total_topics: int = 0

        # Embedding similarity
        self.head_embedding_CD: Tensor = torch.zeros(
            0, self.model_emb.config.hidden_size, device=device
        )  # [num_clusters, hidden_size]

        # NOTE Model, Tokenizer, Device are not saved to the Object to reduce overhead

    # Embedding related functions
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Average-pool the last hidden states of the model."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def final_token_hidden_state(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """Get the hidden state of the final token."""
        return last_hidden_states[
            torch.arange(last_hidden_states.size(0)), attention_mask.sum(dim=1) - 1, :
        ]



    # Adding new topics and deduplication
    def add_new_cluster_head(self, topic: Topic, embedding_D):
        """Add a new cluster head."""
        self.head_topics.append(topic)
        self.num_head_topics += 1
        self.head_embedding_CD = torch.cat((self.head_embedding_CD, embedding_D[None, :]), dim=0)
        self.cluster_topics.append([topic])
        self.cluster_cossims.append([1])
        self.num_topics_per_cluster.append(1)
        return topic

    def append_to_cluster(self, topic: Topic, max_cossim: float, cluster_idx: int) -> Topic:
        """Add a topic to an existing cluster."""
        self.cluster_topics[cluster_idx].append(topic)
        self.cluster_cossims[cluster_idx].append(max_cossim)
        self.num_topics_per_cluster[cluster_idx] += 1
        return topic

    def add_non_refusal_topic(self, topic: Topic):
        """Add a non-refusal topic."""
        self.non_refusal_topics.append(topic)
        self.num_non_refusal_topics += 1
        self.num_total_topics += 1
        return topic




    def incoming_batch(self, topics: List[Topic], device: str, verbose=False) -> List[Topic]:
        """Process a batch of topics to be added to the queue with deduplication."""
        if topics == []:
            print("No topics passed.")
            return

        batch_embeddings_BD = self.batch_embedding(topics, device)

        # deduplicate and add to queue
        # This has to be done one by one to catch duplicates within the incoming batch
        added_head_topics = []
        for topic, embedding_D in zip(topics, batch_embeddings_BD):
            topic = self.process_incoming_topic(topic, embedding_D)
            if topic.is_head:
                added_head_topics.append(topic)

        return added_head_topics

    # Saving, loading and logging
    def to_dict(self):
        """Convert the topic queue to a dictionary representation."""
        topic_dict = {
            "queue": {
                "head_topics": [t.to_dict() for t in self.head_topics],
                "cluster_topics": [
                    [t.to_dict() for t in cluster] for cluster in self.cluster_topics
                ],
                "cluster_cossims": self.cluster_cossims,
            },
            "stats": {
                "num_topics_per_cluster": self.num_topics_per_cluster,
                "num_clusters": self.num_head_topics,
            },
            "config": {
                "cossim_thresh": self.cossim_thresh,
                # TODO: Add model, tokenizer, device
            },
        }
        return topic_dict

    def save(self, path: str):
        """Save the topic queue to a JSON file."""
        topic_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(topic_dict, f)
        return topic_dict

    @classmethod
    def load(cls, topic_dict: dict, model_emb, tokenizer_emb, device: str) -> "TopicQueue":
        """Create a new TopicQueue from a dictionary representation."""
        # Initialize new queue
        queue = cls(model_emb, tokenizer_emb, device)

        # Set basic attributes
        queue.cossim_thresh = topic_dict["cossim_thresh"]
        queue.num_head_topics = topic_dict["num_clusters"]
        queue.num_topics_per_cluster = topic_dict["num_topics_per_cluster"]

        # Reconstruct head topics
        queue.head_topics = [Topic(**topic_data) for topic_data in topic_dict["head_topics"]]

        # Reconstruct cluster topics
        queue.cluster_topics = [
            [Topic(**topic_data) for topic_data in cluster]
            for cluster in topic_dict["cluster_topics"]
        ]

        # Rebuild mean embeddings tensor
        queue.head_embedding_CD = torch.zeros(
            queue.num_head_topics, model_emb.config.hidden_size, device=device
        )

        # Calculate mean embeddings for each cluster
        for i, cluster in enumerate(queue.cluster_topics):
            embeddings = queue.batch_embedding(tokenizer_emb, model_emb, cluster, device)
            queue.head_embedding_CD[i] = embeddings.mean(dim=0)

        return queue

    def __repr__(self) -> str:
        """Return a string representation of the TopicQueue showing stats, config and topics."""
        string = "TopicQueue:\n\n"

        # Stats
        string += "Stats:\n"
        string += f"  Total Clusters: {self.num_head_topics}\n"
        string += f"  Topics per Cluster: {self.num_topics_per_cluster}\n\n"

        # Config
        string += "Config:\n"
        string += f"  Cosine Similarity Threshold: {self.cossim_thresh}\n\n"

        # Topics by cluster
        string += "Clusters:\n"
        for i in range(self.num_head_topics):
            # Head topic
            head = self.head_topics[i]
            string += f"\nCluster {i}:\n"
            string += (
                f"  Head: text='{head.text}', raw='{head.raw}', translation='{head.translation}'\n"
            )

            # Cluster topics
            string += "  Topics:\n"
            for topic in self.cluster_topics[i]:
                if not topic.is_head:  # Skip head topic since we already showed it
                    string += f"    text='{topic.text}', raw='{topic.raw}', translation='{topic.translation}'\n"

        # Total count
        all_topics = sum(len(cluster) for cluster in self.cluster_topics)
        string += f"\nTotal Topics: {all_topics}"

        return string
