from transformers import AutoTokenizer
import torch
from typing import List, Dict


SPECIAL_TOKEN_MAP = {
    "llama" : {
        "names" : ["meta-llama", "allenai", "DeepSeek-R1-Distill-Llama", "perplexity-ai"],
        "token_map" : {
            "BOS": 128000,
            "USER": 128011,
            "ASSISTANT": 128012,
            "NEWLINE": 198,
            "THINK_START": 128013,
            "THINK_END": 128014,
            "EOS": 128001,
        }
    },
    "qwen": {
        "names" : ["Qwen"],
        "token_map" : {
            "BOS": 151646,
            "USER": 151644,
            "ASSISTANT": 151645,
            "NEWLINE": 198,
            "THINK_START": 151648,
            "THINK_END": 151649,
            "EOS": 151643,
        }
    },
}

def get_special_tokens(model_name: str) -> Dict[str, int]:
    """
    Get the special tokens for the model.
    """
    st = None
    for model_type in SPECIAL_TOKEN_MAP.keys():
        match = [name in model_name for name in SPECIAL_TOKEN_MAP[model_type]["names"]]
        if any(match):
            st = SPECIAL_TOKEN_MAP[model_type]["token_map"]
            break
    if st is None:
        raise ValueError(
            f"Unknown model: {model_name}. Model name must contain {SPECIAL_TOKEN_MAP.keys()}"
        )
    return st


def custom_encoding_r1(
    model_name: str,
    tokenizer: AutoTokenizer,
    user_message: str,
    thinking_message: str = "",
    user_suffix: str = "",
    assistant_prefill: str = "",
    force_thought_skip: bool = False,
    template: str = "chat",
) -> List[int]:
    """
    Custom encoding for the model.
    """

    # Verify arguments and get special tokens
    st = get_special_tokens(model_name)

    # Encode user message
    if user_suffix != "":
        user_message = user_message + " " + user_suffix

    user_tokens = tokenizer.encode(user_message, add_special_tokens=False)

    if template == "base":
        # equivalent to calling tokenzer.encode for deepseek-r1-distilled models
        token_ids = [st["BOS"]] + user_tokens
    elif template == "chat":
        # equivalent to calling tokenizer.apply_chat_template for deepseek-r1-distilled models
        token_ids = [st["BOS"]] + [st["USER"]] + user_tokens + [st["ASSISTANT"]]
    else:
        raise ValueError(f"Unknown template: {template}. Choose from 'base' or 'chat'")

    if assistant_prefill != "":
        assert template == "chat", "Assistant prefix is only supported for chat template"
        assistant_prefill_tokens = tokenizer.encode(assistant_prefill, add_special_tokens=False)
        token_ids = token_ids + assistant_prefill_tokens # + [st["THINK_START"]] + [st["NEWLINE"]]

    # Optionally prefill thinking tokens
    if len(thinking_message) > 0:
        thinking_tokens = tokenizer.encode(thinking_message, add_special_tokens=False)
        token_ids = token_ids + [st["THINK_START"]] + [st["NEWLINE"]] + thinking_tokens

    # if assistant_prefill == "" and thinking_message == "":
    #     token_ids = token_ids + [st["THINK_START"]] + [st["NEWLINE"]]
    
    if force_thought_skip:  
        token_ids = token_ids + [st["NEWLINE"]] + [st["THINK_END"]]

    return token_ids


def custom_batch_encoding(
    model_name: str,
    tokenizer: AutoTokenizer,
    user_messages: List[str],
    thinking_message: str = "",
    user_suffix: str = "",
    assistant_prefill: str = "",
    force_thought_skip: bool = False,
    template: str = "chat",
) -> List[int]:
    """
    Custom batch encoding for the model.
    """
    if "deepseek" in model_name:
        token_ids = [
            custom_encoding_r1(
                model_name=model_name,
                tokenizer=tokenizer,
                user_message=user_message,
                thinking_message=thinking_message,
                user_suffix=user_suffix,
                assistant_prefill=assistant_prefill,
                force_thought_skip=force_thought_skip,
                template=template,
            )
            for user_message in user_messages
        ]
        return token_ids
    elif "meta" in model_name:
        token_ids = custom_batch_encoding_Meta(
            tokenizer=tokenizer,
            user_messages=user_messages,
            thinking_message=thinking_message,
            user_suffix=user_suffix,
            assistant_prefill=assistant_prefill,
            force_thought_skip=force_thought_skip,
            template=template,
        )
        return token_ids
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only DeepSeek and Meta models are supported.")

def custom_batch_encoding_Meta(
    tokenizer: AutoTokenizer,
    user_messages: List[str],
    thinking_message: str = "",
    user_suffix: str = "",
    assistant_prefill: str = "",
    force_thought_skip: bool = False,
    template: str = "chat",
) -> List[int]:
    """
    Custom batch encoding for the model.
    """
    if template != "chat":
        raise ValueError(f"Unsupported template: {template}. Only 'chat' template is supported for Meta Llama.")
    if thinking_message != "":
        raise ValueError("Thinking message is not supported for Meta Llama.")
    if force_thought_skip:
        raise ValueError("Forced thought skip is not supported for Meta Llama.")
    
    token_ids = []
    for user_message in user_messages:
        user_message += " " + user_suffix
        if assistant_prefill != "":
            chat_ids = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_prefill},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )[:-1] # Remove assistant EOS token so the assistant keeps generating
        else:
            # Standard template
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=True,
                add_generation_prompt=True,
            )
        token_ids.append(chat_ids)
    return token_ids

def custom_decoding(
    model_name: str,
    tokenizer: AutoTokenizer,
    token_ids_BL: torch.Tensor,
    skip_special_tokens: bool = False,
) -> List[str]:
    """
    Custom decoding for the model.
    """
    st = get_special_tokens(model_name)
    if isinstance(token_ids_BL, torch.Tensor):
        token_ids_BL = token_ids_BL.tolist()
    token_ids = [[id for id in batch if id != st["EOS"]] for batch in token_ids_BL] # Remove padding and EOS tokens
    generated_texts = tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )
    return generated_texts


if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Basic test cases
    user_message = "Hello, how are you?"
    thinking_message = "I am thinking"
    user_suffix = "with urgency"
    assistant_prefill = "Let me think about this."

    print("\n=== Basic Encoding Tests ===")

    base_token_ids = custom_encoding_r1(model_name, tokenizer, user_message, template="base")
    base_token_str = tokenizer.decode(base_token_ids)
    print(f"Base template:\n{base_token_str}\n")

    chat_token_ids = custom_encoding_r1(model_name, tokenizer, user_message, template="chat")
    chat_token_str = tokenizer.decode(chat_token_ids)
    print(f"Chat template:\n{chat_token_str}\n")

    # Test with thinking message
    print("=== With Thinking Message ===")
    chat_thinking_ids = custom_encoding_r1(
        model_name, tokenizer, user_message, thinking_message=thinking_message, template="chat"
    )
    chat_thinking_str = tokenizer.decode(chat_thinking_ids)
    print(f"Chat + thinking:\n{chat_thinking_str}\n")

    # Test with user suffix
    print("=== With User Suffix ===")
    chat_suffix_ids = custom_encoding_r1(
        model_name, tokenizer, user_message, user_suffix=user_suffix, template="chat"
    )
    chat_suffix_str = tokenizer.decode(chat_suffix_ids)
    print(f"Chat + user suffix:\n{chat_suffix_str}\n")

    # Test with assistant prefill
    print("=== With Assistant Prefill ===")
    chat_prefill_ids = custom_encoding_r1(
        model_name, tokenizer, user_message, assistant_prefill=assistant_prefill, template="chat"
    )
    chat_prefill_str = tokenizer.decode(chat_prefill_ids)
    print(f"Chat + assistant prefill:\n{chat_prefill_str}\n")

    # Test batch encoding
    print("=== Batch Encoding Tests ===")
    user_messages = ["What is the weather like?", "Tell me a joke", "How does photosynthesis work?"]

    batch_ids = custom_batch_encoding(model_name, tokenizer, user_messages, template="chat")
    print("Batch encoded messages:")
    for i, ids in enumerate(batch_ids):
        print(f"\nMessage {i + 1}:\n{tokenizer.decode(ids)}")

    # Test combining multiple features
    print("\n=== Combined Features Test ===")
    combined_ids = custom_encoding_r1(
        model_name,
        tokenizer,
        user_message,
        thinking_message=thinking_message,
        user_suffix=user_suffix,
        assistant_prefill=assistant_prefill,
        template="chat",
    )
    combined_str = tokenizer.decode(combined_ids)
    print(f"All features combined:\n{combined_str}")
