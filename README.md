# 🔬 SciQ QA Fine-Tuning with LoRA (PEFT) — CPU-Friendly
This project demonstrates how to fine-tune a pre-trained LLaMA-based model (TinyLlama/TinyLlama-1.1B-Chat-v1.0) using LoRA (Low-Rank Adaptation) for question answering tasks on the allenai/sciq dataset.

## 📦 Dependencies

Install required packages:
```
pip install datasets transformers accelerate peft
```
## 📁 Dataset Used
allenai/sciq

## Format: 
Each entry has:

-  question

-  support (context)

-  correct_answer

-  distractor1, distractor2, distractor3

## 🚀 Training Flow (LoRA)
-  Load Dataset

-  Format Prompt:
```
Context: <support>
Question: <question>
Answer:
```

-  Tokenize & Preprocess

-  Apply LoRA Config

-  Train using Hugging Face Trainer

```
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model = get_peft_model(base_model, lora_config)
```
## 🧠 Inference Example
```
question = "Kilauea in hawaii is the world’s most continuously active volcano..."
support = "Very active volcanoes characteristically eject red-hot rocks and lava..."

prompt = f"Context: {support}\nQuestion: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = peft_model.generate(inputs.input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## 💾 Save & Load the Model
-  To save:
```
peft_model.save_pretrained("./lora-sciq-output")
tokenizer.save_pretrained("./lora-sciq-output")
```
-  To load later:

```
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "./lora-sciq-output")
tokenizer = AutoTokenizer.from_pretrained("./lora-sciq-output")
```
## ⚙️ Training Tips for CPU
> Use select(range(10)) for smaller batches.

> Reduce num_train_epochs to 1.

> Disable saving, logging, and evaluation for faster runs.

## 📌 Sample TrainingArgs

```
training_args = TrainingArguments(
    output_dir="./peft-sciq-cpu-test",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=1e-3,
    save_strategy="no",
    logging_steps=10,
    report_to="none",
    fp16=False,
)
```
## 📈 Example Output
```
Prompt:
Context: Very active volcanoes...
Question: ...rather than this?
Answer:

Model Output:
Answer: smoke and ash
```
## 🤝 Credits
Model: TinyLlama

Dataset: allenai/sciq

Libraries: Hugging Face 🤗, PEFT, LoRA
