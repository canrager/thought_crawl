from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoModel
import torch
import os
import spacy

def load_model(model_name: str, cache_dir: str, device: str):
    """
    Load a huggingface model and tokenizer.

    Args:
        device (str): Device to load the model on ('auto', 'cuda', 'cpu', etc.)
        cache_dir (str, optional): Directory to cache the downloaded model
    """

    # Determine quantization
    torch_dtype = torch.bfloat16
    quantization_config = None

    if "70B" in model_name or "32B" in model_name:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Using 4-bit quantization")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    if "Llama-3.3-70B" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            quantization_config=quantization_config,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            # model_name,
            pretrained_model_name_or_path=os.path.join(cache_dir, model_name),
            torch_dtype=torch_dtype,
            device_map=device,
            quantization_config=quantization_config,
            # cache_dir=cache_dir,
        )

    # Optimize for inference
    model.eval()
    model = torch.compile(model)

    return model, tokenizer

def load_filter_models(cache_dir: str, device: str):
    # Translation LM
    tokenizer_zh_en = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-en", cache_dir=cache_dir
    )
    model_zh_en = AutoModelForSeq2SeqLM.from_pretrained(
        "Helsinki-NLP/opus-mt-zh-en", cache_dir=cache_dir, device_map=device
    )

    # Text embedding for measuring semantic similarity
    tokenizer_emb = AutoTokenizer.from_pretrained(
        "intfloat/multilingual-e5-large-instruct", cache_dir=cache_dir
    )
    model_emb = AutoModel.from_pretrained(
        "intfloat/multilingual-e5-large-instruct",
        cache_dir=cache_dir,
        device_map=device,
        # torch_dtype=torch.bfloat16,
    )

    # NLP model for formatting / subject extraction
    # needs python -m spacy download en_core_web_sm
    model_spacy_en = spacy.load("en_core_web_sm")

    return {
        "tokenizer_zh_en": tokenizer_zh_en,
        "model_zh_en": model_zh_en,
        "tokenizer_emb": tokenizer_emb,
        "model_emb": model_emb,
        "model_spacy_en": model_spacy_en,
    }
