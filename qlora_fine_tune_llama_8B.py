from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os

# Set environment variable to disable upper limit for memory allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Load base model without 4-bit quantization
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load in 16-bit precision instead of 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # Remove or comment out torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# First, identify the correct target modules by examining model architecture
# This will help us debug the issue
module_names = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
        module_names.append(name)
        
print("Available modules for LoRA fine-tuning:")
print(module_names[:20])  # Print first 20 to avoid excessive output

# Using common Linear layers in transformer models
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # Updated target modules based on actual model architecture
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Common transformer attention module names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
print("Trainable parameters:")
model.print_trainable_parameters()

# Load dataset
from datasets import load_dataset

dataset = load_dataset("json", data_files="/Users/tgusain/Desktop/llama/pylang_1K_tokens.jsonl")

# Tokenization function
def tokenize(example):
    # Add chat template formatting
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": example["input"]}], tokenize=False)
    tokens = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"].copy()
    }

dataset = dataset.map(tokenize, remove_columns=["input"])

# Training Configuration
from transformers import TrainingArguments, Trainer

# And later in the TrainingArguments
training_args = TrainingArguments(
    output_dir="./output/llama-ft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=False,  # Disable FP16
    optim="adamw_torch",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=lambda data: {
        'input_ids': torch.tensor([f['input_ids'] for f in data], dtype=torch.long),
        'attention_mask': torch.tensor([f['attention_mask'] for f in data], dtype=torch.long),
        'labels': torch.tensor([f['labels'] for f in data], dtype=torch.long)
    }
)

trainer.train()

# Save fine-tuned model and tokenizer
peft_model_path = "./output/llama-ft-final"
model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# Test the model
# Load the saved model for inference
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto", 
    trust_remote_code=True
)

# Load PEFT adapter
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path, trust_remote_code=True)

# Generate text
test_prompt = "Summarize: Your document text"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": test_prompt}], return_tensors="pt").to(peft_model.device)

with torch.no_grad():
    outputs = peft_model.generate(inputs, max_length=200, do_sample=True, temperature=0.7)
    
print(tokenizer.decode(outputs[0], skip_special_tokens=True))