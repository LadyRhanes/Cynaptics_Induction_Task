from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_WITHOUT_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

# ─────────────────────────────────────────────
# FORMATTERS
# ─────────────────────────────────────────────
def format_input(example):
    if example.get("input") and example["input"].strip():
        return PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        ).rstrip()
    else:
        return PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
        ).rstrip()


def format_alpaca_prompt(example):
    if example.get("input") and example["input"].strip():
        text = PROMPT_WITH_INPUT.format(
            instruction=example["instruction"],
            input=example["input"],
        ) + example["output"]
    else:
        text = PROMPT_WITHOUT_INPUT.format(
            instruction=example["instruction"],
        ) + example["output"]

    return {"text": text}

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def load_alpaca_dataset(split="train", test_size=0.1, val_size=0.1, seed=42):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.map(format_alpaca_prompt)

    train_val, test_ds = dataset.train_test_split(
        test_size=test_size, seed=seed
    ).values()

    val_ratio = val_size / (1.0 - test_size)
    train_ds, val_ds = train_val.train_test_split(
        test_size=val_ratio, seed=seed
    ).values()

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    return splits[split]

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
train_data = load_alpaca_dataset("train")
val_data   = load_alpaca_dataset("val")
test_data  = load_alpaca_dataset("test")

# ─────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.encoded_texts = []

        for entry in data:
            prompt = format_input(entry)
            full_text = prompt + entry["output"]

            tokens = tokenizer.encode(
                full_text,
                truncation=True,
                max_length=max_length
            )

            self.encoded_texts.append(tokens)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)

# ─────────────────────────────────────────────
# COLLATE
# ─────────────────────────────────────────────
def custom_collate_fn(batch, pad_token_id, ignore_index=-100,
                      allowed_max_length=256, device="cpu"):

    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs  = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        targets[mask] = ignore_index

        inputs_lst.append(inputs[:allowed_max_length])
        targets_lst.append(targets[:allowed_max_length])

    return (
        torch.stack(inputs_lst).to(device),
        torch.stack(targets_lst).to(device)
    )

# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────
collate = partial(
    custom_collate_fn,
    pad_token_id=tokenizer.pad_token_id,
    device=device
)

train_loader = DataLoader(
    InstructionDataset(train_data, tokenizer),
    batch_size=2,
    shuffle=True,
    drop_last=True,
    collate_fn=collate
)

val_loader = DataLoader(
    InstructionDataset(val_data, tokenizer),
    batch_size=2,
    shuffle=False,
    collate_fn=collate
)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.to(device)

# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
def calc_loss_batch(input_batch, target_batch):
    logits = model(input_batch).logits
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1),
        ignore_index=-100
    )

# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def generate(model, idx, max_new_tokens, context_size, eos_id):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond).logits

        logits = logits[:, -1, :] / 0.8
        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, 1)

        if (next_id == eos_id).all():
            break

        idx = torch.cat([idx, next_id], dim=1)

    return idx

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(3):
    model.train()
    losses = []

    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        loss = calc_loss_batch(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % 200 == 0 and step > 0:
            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch} | Step {step} | Avg Loss: {avg_loss:.3f}")
            losses = []

    # VALIDATION
    model.eval()
    val_losses = []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_loss = calc_loss_batch(x_val, y_val)
            val_losses.append(val_loss.item())

    print(f"\nEpoch {epoch} | Validation Loss: {sum(val_losses)/len(val_losses):.3f}\n")
    model.train()

# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
model.eval()

for entry in test_data.select(range(3)):
    input_text = format_input(entry)

    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=200,
        context_size=256,
        eos_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(output_ids[0])
    response = generated[len(input_text):].strip()

    print("\nINPUT:\n", input_text)
    print("\nEXPECTED:\n", entry["output"])
    print("\nMODEL:\n", response)
    print("="*50)
