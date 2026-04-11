# GPT-2 Fine-Tuning on Alpaca Dataset (Single Script)

## Overview

This project fine-tunes a pretrained GPT-2 (124M) model on the Alpaca instruction-following dataset using supervised fine-tuning (SFT).

The goal is to adapt a general language model into an instruction-following assistant that can generate responses based on a given prompt format.

---

## Approach

### 1. Pretrained Model

We use the Hugging Face model:

* `openai-community/gpt2`

This model already understands language, so we only need to fine-tune it for instruction-following.

---

### 2. Dataset

We use:

* `tatsu-lab/alpaca` (~52K examples)

Each example contains:

* `instruction`
* `input` (optional)
* `output` (target response)

---

### 3. Prompt Formatting

Each example is converted into a structured prompt.

#### With input:

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

#### Without input:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

The model is trained to generate the `output` after this prompt.

---

### 4. Tokenization

* Tokenizer: GPT-2 tokenizer
* Padding token is set to EOS token
* Sequences are truncated to `max_length = 256`

---

### 5. Training Setup

* Loss: Cross-Entropy Loss
* Ignore padding tokens using `ignore_index = -100`
* Optimizer: AdamW
* Learning rate: `3e-5`
* Gradient clipping: `1.0`
* Batch size: `2`
* Epochs: `3`

---

### 6. Data Splitting

Dataset is split into:

* Train
* Validation
* Test

Using:

* `test_size = 0.1`
* `val_size = 0.1`

---

### 7. Training Objective

The model is trained as a **causal language model**, meaning:

* Input tokens predict the next token
* Targets are shifted versions of inputs

---

### 8. Inference

During inference:

* Only the prompt (without output) is given
* The model generates tokens autoregressively
* Sampling is used (temperature = 0.8)

---

## How to Run

### 1. Install dependencies

```
pip install torch transformers datasets
```

### 2. Run the script

```
python fine_tuning.py
```

---

## Sample Outputs

### Example 1

**Instruction:**
What would be the best type of exercise for a person who has arthritis?

**Model Output:**
Suggests yoga, Pilates, and low-impact exercises (partially correct but repetitive).

---

### Example 2

**Instruction:**
Calculate the atomic mass for lithium.

**Model Output:**
Incorrect numerical reasoning (hallucination observed).

---

### Example 3

**Instruction:**
Convert binary to ASCII

**Model Output:**
Fails to properly decode (task-specific weakness).

---

## Observations

### What Works

* Model learns general response structure
* Produces fluent and grammatically correct text
* Understands instruction format

### Limitations

* Hallucinations in factual/numeric tasks
* Repetition in responses
* Weak performance on precise transformations (e.g., binary → ASCII)

---

## Key Learnings

* Fine-tuning improves instruction-following but does not guarantee correctness
* GPT-2 lacks strong reasoning ability compared to modern instruction-tuned models
* Proper data formatting is critical for SFT
* Loss decreasing does not always mean better factual accuracy

---

## Possible Improvements

* Increase context length (512 or 1024)
* Train for more epochs
* Use better decoding (top-k / top-p sampling)
* Use a larger base model (GPT-2 medium or larger)
* Apply instruction masking (train only on response tokens)

---

## Conclusion

This project demonstrates how a pretrained language model can be adapted to a downstream task using supervised fine-tuning.

While the model learns the structure of instruction-following, it still struggles with factual accuracy and reasoning, highlighting the limitations of small language models like GPT-2.

---

