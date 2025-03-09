from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login

# Authenticate with Hugging Face
login(token="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")

model_name = "Qwen/Qwen2.5-3B-Instruct"  # Change the model name to gpt2

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

try:
    model = AutoModelForCausalLM.from_pretrained(model_name) #, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except EnvironmentError as e:
    print(f"Error accessing the model: {e}")
    print("Make sure you have access to the model at https://huggingface.co/{model_name}")
    exit(1)

from datasets import load_dataset

dataset = load_dataset("json", data_files="/Users/tgusain/Desktop/llama/dummy_data.jsonl")

def tokenize(example):
    example["input_ids"] = tokenizer(example["input"], truncation=True, padding="max_length", max_length=4096)["input_ids"]
    return example

dataset = dataset.map(tokenize)