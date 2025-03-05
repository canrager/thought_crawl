from typing import List, Optional
from tqdm import trange

import nnsight
from nnsight import LanguageModel
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from core.tokenization_utils import custom_decoding, custom_batch_encoding


def single_generate_from_tokens(
    model,
    tokenizer,
    input_ids,
    max_generated_tokens,
    skip_special_tokens=False,
    temperature: Optional[float] = None,
):
    """
    Generate text based on the input prompt.

    Args:
        prompt (str): Input text prompt
        model: The loaded model
        tokenizer: The loaded tokenizer
        max_length (int): Maximum length of generated text

    Returns:
        str: Generated text
    """

    # Prepare input ids
    input_ids = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
        top_p = temperature
    else:
        do_sample = False
        top_p = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_generated_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
    return generated_text


def custom_pad(input_ids, tokenizer):
    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad input_ids to the same length
    max_length = max(len(ids) for ids in input_ids)
    padded_input_ids = [
        [tokenizer.pad_token_id] * (max_length - len(ids)) + ids for ids in input_ids
    ]
    padded_attention_mask = [[0] * (max_length - len(ids)) + [1] * len(ids) for ids in input_ids]
    return padded_input_ids, padded_attention_mask


def batch_generate_from_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
):
    """
    Generate text based on the input prompt.
    Note, we assume tokenized input_ids were generated with special tokens.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        input_ids: The input ids
        max_length: The maximum length of the generated text
        skip_special_tokens: Whether to skip the special tokens
    """

    padded_input_ids_BL, padded_attention_mask_BL = custom_pad(input_ids_BL, tokenizer)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids_BL).to(model.device)
    attention_mask_tensor = torch.tensor(padded_attention_mask_BL).to(model.device)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
        top_p = temperature
    else:
        do_sample = False
        top_p = None

    with torch.no_grad():
        outputs = model.generate(
            input_ids_tensor,
            attention_mask=attention_mask_tensor,
            max_length=max_generation_length,  # Use max_generation_length instead of max_length
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    model_name = model.config._name_or_path
    generated_texts = custom_decoding(model_name, tokenizer, outputs, skip_special_tokens)

    return generated_texts


def generate_text_from_tokens_NDIF(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
):
    # Generate text
    padded_input_ids_BL, padded_attention_mask_BL = custom_pad(input_ids_BL, tokenizer)
    padded_input_ids_BL = torch.tensor(padded_input_ids_BL)
    padded_attention_mask_BL = torch.tensor(padded_attention_mask_BL)

    # Set sampling parameters
    if temperature is not None:
        do_sample = True
        top_p = temperature
    else:
        do_sample = False
        top_p = None

    # Generate text
    with model.generate(
        {"input_ids": padded_input_ids_BL, "attention_mask": padded_attention_mask_BL},
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True,  # Run the model remotely on NDIF
    ):
        outputs = nnsight.list().save()

        with model.lm_head.all():
            outputs.append(model.lm_head.output[0][-1].argmax(dim=-1))

    # Decode and return the generated text
    outputs = outputs.value
    in_out = torch.cat([torch.tensor(padded_input_ids_BL), outputs], dim=1)
    in_out_tokens = custom_decoding(model_name, tokenizer, in_out, skip_special_tokens)
    return in_out_tokens


def batch_generate(
    model,
    tokenizer,
    selected_topics: List[str],
    user_message_template: str = "{}",
    assistant_prefill: str = "",
    thinking_message: str = "",
    force_thought_skip: bool = False,
    tokenization_template: str = "chat",
    num_samples_per_topic: int = 1,
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
    skip_special_tokens: bool = False,
    remote: bool = False,
) -> List[str]:
    """Given a list of seed topics, return a list of raw model generations,
    self.config.num_samples_per_topic determines number of generations for identical input topics.
    """
    generated_texts = []
    model_name = model.config._name_or_path

    user_messages = [user_message_template.format(topic) for topic in selected_topics]
    input_ids = custom_batch_encoding(
        model_name=model_name,
        tokenizer=tokenizer,
        user_messages=user_messages,
        assistant_prefill=assistant_prefill,
        thinking_message=thinking_message,
        force_thought_skip=force_thought_skip,
        template=tokenization_template,
    )

    if remote:
        for _ in range(num_samples_per_topic):
            batch_generations = generate_text_from_tokens_NDIF(
                model=model,
                tokenizer=tokenizer,
                input_ids_BL=input_ids,
                max_new_tokens=max_new_tokens,
                max_generation_length=None,
                temperature=temperature,
                skip_special_tokens=skip_special_tokens,
            )
            generated_texts.extend(batch_generations)
    else:
        with torch.inference_mode(), torch.autocast(device_type=model.device):
            for _ in range(num_samples_per_topic):
                batch_generations = batch_generate_from_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_BL=input_ids,
                    max_generation_length=None,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    skip_special_tokens=skip_special_tokens,
                )
                generated_texts.extend(batch_generations)

    if verbose:
        input_tokens = custom_decoding(model_name, tokenizer, input_ids, skip_special_tokens)
        for i, o in zip(input_tokens, generated_texts):
            print(f"input: {i}\noutput: {o}\n\n")
    return generated_texts


# Embedding related functions
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Average-pool the last hidden states of the model."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def final_token_hidden_state(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Get the hidden state of the final token."""
    return last_hidden_states[
        torch.arange(last_hidden_states.size(0)), attention_mask.sum(dim=1) - 1, :
    ]


def compute_embeddings(
    tokenizer_emb: AutoTokenizer, model_emb: AutoModel, words: List[str], prefix: str = "query: "
) -> Tensor:
    """Embed a batch of words."""
    batch_formatted = [f"{prefix}{word.lower()}" for word in words]
    batch_dict = tokenizer_emb(batch_formatted, padding=True, truncation=False, return_tensors="pt")
    batch_dict = batch_dict.to(model_emb.device)

    # outputs.last_hidden_state.shape (batch_size, sequence_length, hidden_size) hidden activations of the last layer
    outputs = model_emb(**batch_dict)
    batch_embeddings_BD = final_token_hidden_state(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )
    batch_embeddings_BD = F.normalize(batch_embeddings_BD, p=2, dim=1)

    return batch_embeddings_BD


def batch_compute_embeddings(
    tokenizer_emb: AutoTokenizer,
    model_emb: AutoModel,
    words: List[str],
    batch_size: int = 100,
    prefix: str = "query: ",
) -> Tensor:
    """Embed a batch of words."""
    batch_embeddings_BD = []
    for i in trange(0, len(words), batch_size):
        batch_words = words[i : i + batch_size]
        batch_embeddings_BD.append(
            compute_embeddings(tokenizer_emb, model_emb, batch_words, prefix=prefix)
        )
    batch_embeddings_BD = torch.cat(batch_embeddings_BD, dim=0)
    return batch_embeddings_BD


def compute_openai_embeddings(
    openai_client, 
    model_name: str, 
    words: List[str], 
    prefix: str = "query: "
) -> Tensor:
    """Embed a batch of words using OpenAI API."""
    batch_formatted = [f"{prefix}{word.lower()}" for word in words]
    
    response = openai_client.embeddings.create(
        input=batch_formatted,
        model=model_name
    )
    
    # Extract embeddings from response
    embeddings = [data.embedding for data in response.data]
    
    # Convert to tensor
    batch_embeddings_BD = torch.tensor(embeddings, dtype=torch.float)
    
    # OpenAI embeddings are already normalized, but normalize again just to be sure
    batch_embeddings_BD = F.normalize(batch_embeddings_BD, p=2, dim=1)
    
    return batch_embeddings_BD


def batch_compute_openai_embeddings(
    openai_client,
    model_name: str,
    words: List[str],
    batch_size: int = 100,
    prefix: str = "query: ",
) -> Tensor:
    """Embed a batch of words using OpenAI API."""
    batch_embeddings_BD = []
    for i in trange(0, len(words), batch_size):
        batch_words = words[i : i + batch_size]
        batch_embeddings_BD.append(
            compute_openai_embeddings(openai_client, model_name, batch_words, prefix=prefix)
        )
    batch_embeddings_BD = torch.cat(batch_embeddings_BD, dim=0)
    return batch_embeddings_BD


if __name__ == "__main__":
    from nnsight import CONFIG
    from core.project_config import INPUT_DIR

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    model = LanguageModel(model_name)
    tokenizer = model.tokenizer

    with open(INPUT_DIR / "nns.txt", "r") as f:
        NNS = f.read()
    CONFIG.API.APIKEY = NNS.strip()
    CONFIG.APP.REMOTE_LOGGING = False

    TOPICS = ["capital of France", "main city of Germany", "most populous city in Italy"]
    USER_MESSAGE_TEMPLATE = "What is the {}?"
    THINKING_MESSAGE = "I know that."

    generated_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        selected_topics=TOPICS,
        user_message_template=USER_MESSAGE_TEMPLATE,
        thinking_message=THINKING_MESSAGE,
        max_new_tokens=100,
        tokenization_template="chat",
        remote=True,
    )
    print(generated_texts)