## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install torch requests
```

---

### 2. Run Training + Generation

```bash
python task1_sol.py
```

This script will:

* Download the dataset (if not already present)
* Train the model
* Generate text after training completes

---

## 📂 Dataset

* **Name:** Tiny Shakespeare
* Automatically downloaded from:

  ```
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  ```
* Saved locally as:

  ```
  shakespeare.txt
  ```

---

## ⚙️ Model Hyperparameters

| Hyperparameter   | Value           |
| ---------------- | --------------- |
| Vocabulary Type  | Character-level |
| Embedding Size   | 128             |
| Number of Layers | 4               |
| Number of Heads  | 4               |
| Context Length   | 128             |
| Batch Size       | 64              |
| Dropout          | 0.2             |
| Learning Rate    | 3e-4            |
| Training Steps   | 40,000          |

### Explanation

* **Character-level vocab**: Each character is treated as a token
* **Embedding Size**: Size of token representations
* **Layers**: Number of Transformer blocks
* **Heads**: Multi-head attention splits learning across multiple heads
* **Context Length**: Number of previous tokens the model considers
* **Dropout**: Helps prevent overfitting
* **Optimizer**: AdamW for stable training

---

## 🧠 Model Architecture

The model consists of:

* Token Embeddings + Positional Embeddings
* Transformer Blocks:

  * Multi-Head Self-Attention
  * Feedforward Network
  * Residual Connections + Layer Normalization
* Final Linear Layer for next-token prediction

---

## ✨ Text Generation

* Autoregressive generation
* Temperature sampling (default = 0.8)

---

## 📝 Sample Output

```
unto be angrown extremest how the benefit:
Thou shut me, for excelles your son are their by,
And infect to first.

Second Lord:
I will rish your well and in my arms,
The heavens of a love that spirit's man be as dosses
Of Yorkissing to take the dear mine eyes.
Like his head down, with the blood, the thire
With many obind him to son, we were craves for
His own gobering: call'd hath rest the right.

ANGELO:
Immittain.

ISABELLAND:
Your patient soul the next of the earth, I do will
Her frant which
```

### Observations

* Learns dialogue structure (character names, formatting)
* Mimics Shakespearean tone
* Generates plausible but sometimes incorrect words

---

## 📁 Project Structure

```
├── task1_sol.py        # Training + generation script
├── shakespeare.txt     # Dataset (auto-downloaded)
├── README.md
```

---

## ⚠️ Notes

* Training on CPU can be slow — GPU is recommended
* Reduce `batch_size` if you encounter memory issues
* Longer training improves output quality

---

## 🙌 Acknowledgements

Inspired by transformer-based language models and educational implementations of GPT and Adrej Ktharpy.
