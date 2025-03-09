# Fine-Tuning LLaMA 3 7B (4-bit QLoRA) on M3 Max (MacBook)

## **Overview**
This guide will walk you through setting up and fine-tuning **LLaMA 3 7B (4-bit QLoRA)** on an **M3 Max MacBook (64GB RAM)** for your private documents. Since M3 Max lacks CUDA support, we will use **Metal Performance Shaders (MPS)** for acceleration.

---

## **Prerequisites**
### **1. Install Required Packages**
Ensure you have Python installed (preferably **Python 3.10+**). Then, install the required libraries:
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers peft bitsandbytes accelerate datasets sentencepiece safetensors
pip3 install llama-cpp-python
```

### **2. Enable Metal Backend for PyTorch**
Torch will automatically use Metal for acceleration. You can verify this with:
```python
import torch
print(torch.backends.mps.is_available())  # Should return True
```

---

## **Step 1: Load LLaMA 3 7B in 4-bit QLoRA**
We use **Hugging Face's Transformers library** to load the model in **4-bit mode** to fit within memory.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## **Step 2: Prepare Your Dataset**
Format your private documents in **JSONL (JSON Lines) format**:
```jsonl
{"instruction": "Summarize this document", "input": "Full text of your document here", "output": "Expected summary"}
```

Use `datasets` to load and tokenize:
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="your_data.jsonl")

def tokenize(example):
    example["input_ids"] = tokenizer(example["input"], truncation=True, padding="max_length", max_length=4096)["input_ids"]
    return example

dataset = dataset.map(tokenize)
```

---

## **Step 3: Fine-Tune LLaMA 3 7B with QLoRA**
Use **PEFT (Parameter Efficient Fine-Tuning)** to train using **QLoRA**.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**Training Configuration:**
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llama3-7b-finetuned",
    per_device_train_batch_size=1,  # MacBooks have limited RAM
    gradient_accumulation_steps=8,  # Simulate larger batch size
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    optim="adamw_torch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
```

---

## **Step 4: Save and Use the Fine-Tuned Model**
Once training is complete, save the fine-tuned model:
```python
model.save_pretrained("./fine-tuned-llama3-7b")
tokenizer.save_pretrained("./fine-tuned-llama3-7b")
```

To test the model:
```python
from transformers import pipeline

generator = pipeline("text-generation", model="./fine-tuned-llama3-7b", tokenizer=tokenizer)
print(generator("Summarize: Your document text", max_length=200))
```

---

## **Performance Notes**
- Fine-tuning **10 docs (~10K tokens each) takes ~1-3 hours** on M3 Max.
- **Inference will work fine** but expect some delay (~5 sec per response).
- Use **smaller models (Mistral 7B, TinyLLaMA)** for faster training.

---

## **Optimizations for Faster Fine-Tuning**
1. **Reduce sequence length** → Train on **shorter chunks (2048 tokens instead of 4096)**.
2. **Use FP16** → Run training in **mixed precision (half-precision math)**.
3. **Use fewer epochs** → Start with **1 epoch and increase if needed**.
