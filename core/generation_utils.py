from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from core.project_config import DEVICE, CACHE_DIR
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
    input_ids = torch.tensor([input_ids], device=DEVICE)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)

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

    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pad input_ids to the same length
    max_length = max(len(ids) for ids in input_ids_BL)
    padded_input_ids_BL = [
        [tokenizer.pad_token_id] * (max_length - len(ids)) + ids for ids in input_ids_BL
    ]
    padded_attention_mask_BL = [
        [0] * (max_length - len(ids)) + [1] * len(ids) for ids in input_ids_BL
    ]

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids_BL).to(DEVICE)
    attention_mask_tensor = torch.tensor(padded_attention_mask_BL).to(DEVICE)

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


def batch_generate(
    model,
    tokenizer,
    selected_topics: List[str],
    user_message_template: str,
    assistant_prefill: str = "",
    thinking_message: str = "",
    force_thought_skip: bool = False,
    tokenization_template: str = "chat",
    num_samples_per_topic: int = 1,
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
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

    with torch.inference_mode(), torch.autocast(device_type=DEVICE):
        for _ in range(num_samples_per_topic):
            batch_generations = batch_generate_from_tokens(
                model=model,
                tokenizer=tokenizer,
                input_ids_BL=input_ids,
                max_generation_length=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            generated_texts.extend(batch_generations)

    return generated_texts


if __name__ == "__main__":
    from core.model_utils import load_model

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model, tokenizer = load_model(model_name, device=DEVICE, cache_dir=CACHE_DIR)

    user_messages = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]
    thinking_messages = ["I know that.", "True.", "I forgot how long ago that was."]

    generated_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        user_messages=user_messages,
        thinking_messages=thinking_messages,
        max_new_tokens=100,
        model_name=model_name,
        prompt_template="base",
    )
    print(generated_texts)
